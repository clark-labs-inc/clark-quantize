# Qwen3.5-9B Native-Context Clark Air QAT Proposal

## Executive Summary

Goal: train and evaluate Clark Air compression on Qwen3.5-9B at the model's native operating envelope:

- input context: up to 1,000,000 tokens
- output budget: up to 32,000 generated tokens
- modalities: text, image, video, and mixed multimodal records
- compression target: first win behavior-preserving all-2D ternary, then revisit binary or mixed rate

The central change from the previous experiments is that the training/evaluation target must be long-horizon behavior, not only short-context teacher-cache KL. A model that improves KL on 1k-token dense rows can still collapse in free generation. The native-context experiment therefore needs both:

1. bulk hard-forward dense distillation for codebook learning, and
2. native-context survival distillation on 1M-token prefixes and 32k-token generations.

The current evidence says the hard-forward trainer executes on one A100 with a
low-state optimizer, but the update path is not yet proven:

- all-2D QAT with `bonsai_qat8` OOMs on one 80GB A100;
- all-2D hard-forward QAT with `factored_adafactor` fits;
- a 512-row dense top-64+other cache with 260,490 supervised positions builds correctly;
- a 128-step hard run improves some training-stream rows but remains generation-collapsed;
- forensics after 128 steps show almost no codebook movement;
- that 128-step result used BF16 masters, so it is a bug report, not a valid
  slope baseline.

Therefore, this proposal is not "run the same 1k cache longer." It is a staged native-context program that measures whether long hard-forward training plus native-context on-policy data can move the residual stream into a codebook-native basis.

## Immediate Corrections Before Launch

The next run must not build weeks of evidence on top of the 128-step artifact.
Fix and gate in this order:

1. Run the existing hard checkpoint with `embed_tokens` and `lm_head` left BF16.
   This separates "body cannot learn ternary" from "head/embed ternary strangles
   top1 and format survival."
2. Switch training masters to FP32, or add a proven stochastic-rounding/Kahan
   compensation path for BF16 masters. The current proposal chooses FP32 masters
   for the slope run.
3. Run a 200-step integration test that logs sampled
   `mean_abs(u_t - u_0)`, `rms(u_t - u_0)`, moved fraction, global grad norm,
   and effective clip scale. If sampled masters do not move at the expected
   LR-times-steps scale, stop.
4. Rerun the 128-step hard probe only after the master-update path is proven.
   Healthy forensics should show nontrivial near-boundary code changes, roughly
   in the `1e-3` to `1e-2` range for code-changed fraction on a short probe,
   concentrated near quantization boundaries. Near-zero movement means the run
   is still invalid.
5. Run a 2k-5k step LR sweep over `{2e-5, 1e-4, 5e-4}` with 500-1k warmup.
   Only the winning LR should receive the 100k-step dense slope budget.
6. Run paired dense-slope arms:
   - all-2D ternary, including embeddings/head;
   - ternary body with BF16 embeddings/head.
7. Build only dense 8k shards and about 100 native 64k records before the LR
   sweep. Gate 256k and 1M cache builds on Stage 1 passing.

## Scientific Hypothesis

Compression target:

```text
Qwen3.5-9B BF16 behavior
  ~= Clark Air all-2D ternary behavior
```

under native contexts:

```text
input length <= 1M tokens
output length <= 32k tokens
```

The hypothesis:

```text
Behavior preservation at ~2.125 bpw requires global hard-forward training
on a large token stream, plus long-context on-policy survival distillation.
```

The negative hypothesis:

```text
Short-context sparse teacher-cache KL, local MLP/block patching, or alpha-ramp
QAT will not recover native-context generation.
```

The Prism/Bonsai-style interpretation is that successful low-bit LLMs are trained into a new codebook-native residual basis. We should expect sign/support/scale movement over a large token budget, not instant lossless behavior from a small cache.

## Target Format

First artifact:

```text
all 2D tensors: ternary g128
scales: FP16 per group
1D norms: continuous BF16/FP16
embedding: ternary
lm_head: ternary
attention projections: ternary
linear-attention projections: ternary
MLP gate/up/down: ternary
```

Optimization control:

```text
control arm: ternary body + BF16 embed_tokens + BF16 lm_head
final target: ternary embed_tokens + ternary lm_head
```

If the control arm learns while the all-ternary arm does not, do not conclude
that the body cannot learn. Conclude that head/embed quantization must be
staged: train the body against a BF16 lexical boundary first, then run a
head/embed-focused KD phase before packing the final strict ternary artifact.

Physical storage for 2D weights:

```text
2 code bits + 16 scale bits / 128 weights = 2.125 bpw
16 / 2.125 = 7.53x compression on quantized 2D tensors
```

Do not start with:

- binary,
- mixed binary/ternary,
- FP8 rescue islands,
- LoRA/QDoRA adapters,
- alpha ramp,
- local MLP-only patches.

Those are later options if all-2D ternary proves the behavior-recovery curve but cannot hit final quality.

Cheap conditioning arm:

```text
PTQ init only: Hadamard/QuaRot-style incoherence preprocessing folded into
adjacent 2D tensors, then the same ternary g128 projection and dense slope gate.
```

This arm is allowed because it changes the initialization basin without adding
adapters or changing the final low-bit format. It should be compared against the
plain PTQ initialization before committing the long dense run.

## Model And Runtime Requirements

### Baseline Must Be Proven First

Before compressed training, prove that the BF16 model can actually run the target envelope in our environment:

```text
Qwen3.5-9B BF16
1M-token input prefill
32k-token output generation
same tokenizer/chat template
same processor payloads for image/video
same stop/format contract
```

Preflight gates:

- 1M text-only prompt prefill succeeds.
- 1M mixed document/code prompt prefill succeeds.
- multimodal prompt with image/video payloads and long text succeeds.
- 32k deterministic generation does not terminate prematurely under BF16.
- AgentWorld-style format remains valid under BF16 at long output.

If BF16 cannot pass the native-context runtime gate, the compressed experiment is invalid.

### Hardware

One 80GB A100 can run short hard-forward QAT with `factored_adafactor`, but not full native-context backprop.

Recommended serious run:

```text
8x A100 80GB or better
FSDP/ZeRO for masters + optimizer
paged KV cache for generation/eval
NVMe for teacher cache shards
```

One-A100 role:

- build/cache data shards,
- smoke dense KD,
- run low-state slope probes,
- run short generation gates,
- debug loaders and packers.

It should not be expected to complete the 1M/32k training program by itself.

## Data Design

We need two different datasets.

### Dataset A: Bulk Codebook Distillation

Purpose: move weights into a low-bit codebook-native solution.

This should be pretraining-like and high-token-count:

- web text,
- code,
- math/reasoning,
- multilingual,
- instruction/chat,
- multimodal image QA,
- video QA/captioning.

Recommended first serious budget:

```text
minimum curve: 1B-2B supervised tokens
credible run: 10B+ supervised tokens
strong run: 30B-100B supervised tokens
```

The 10k-row/260k-token dense cache is a plumbing test only. It cannot establish lossless compression.

### Dataset B: Native-Context Survival

Purpose: ensure the compressed model operates at 1M input and 32k output.

Records should be generated as long scenarios rather than random concatenations only.

#### Text Long-Context Records

Examples:

- multi-document QA over 1M-token corpora,
- books or long reports with delayed questions,
- legal/financial/technical corpora with citations,
- long multilingual corpora requiring cross-lingual retrieval,
- codebase reasoning over many files,
- synthetic needle-in-haystack with natural distractors.

Record shape:

```json
{
  "id": "long_text_000001",
  "modalities": ["text"],
  "input_tokens_target": 1000000,
  "output_tokens_target": 4096,
  "messages": [...],
  "answer_requirements": [...],
  "supervision_windows": [...]
}
```

#### Image Long-Context Records

Examples:

- many-page document images with text questions,
- chart/table image sets embedded in long context,
- scientific figure collections,
- UI screenshots over a long workflow trace.

Record shape:

```json
{
  "id": "image_long_000001",
  "modalities": ["text", "image"],
  "images": [...],
  "text_context": "...",
  "question": "...",
  "answer": "...",
  "processor_payload_policy": {
    "image_longest_edge": 65536,
    "image_shortest_edge": 4096
  }
}
```

#### Video Long-Context Records

Examples:

- long video QA with sampled frames,
- temporal ordering questions,
- multi-video comparison,
- action/event summaries with 32k output.

Record shape:

```json
{
  "id": "video_long_000001",
  "modalities": ["text", "video"],
  "video_frames": [...],
  "frame_sampling": "uniform+event",
  "question": "...",
  "answer": "...",
  "temporal_requirements": [...]
}
```

For video, we should not assume a raw continuous video can be naively placed into a 1M token context. The processor converts frames/video into model tensors and effective visual tokens. The cache must store the processor payload or reproducible frame references, not just decoded text.

## Teacher Cache Format

Caching full logits for 1M input positions is not practical.

Use two cache modes:

### Dense Short/Medium Mode

For codebook learning:

```text
context length: 1k-32k
supervision: every position
teacher labels: top-k logits + other bucket
top-k: 64 or 128
```

This is the most token-efficient distillation mode.

### Native Sparse/Windowed Mode

For 1M contexts:

Do not cache every position. Cache:

- final answer positions,
- format-critical positions,
- retrieval target positions,
- high-entropy teacher positions,
- periodic anchor positions every N tokens,
- dense windows around important spans,
- windows around student failure points.

Suggested 1M record supervision:

```text
periodic anchors: every 2048 or 4096 tokens
dense local windows: 2k-8k tokens around target evidence
answer/output tokens: dense
format tokens: dense
failure windows: dense after on-policy generation
```

Cache row should include:

```json
{
  "format": "qwen_native_context_teacher_cache_v1",
  "model_dir": "/workspace/models/Qwen3.5-9B",
  "record_id": "...",
  "modalities": ["text", "image", "video"],
  "input_token_count": 1000000,
  "output_token_count": 32768,
  "processor_payload_refs": {...},
  "supervision": [
    {
      "window_id": "answer_000",
      "start_token": 998000,
      "end_token": 1000000,
      "top_values": "...",
      "top_indices": "...",
      "teacher_logsumexp": "..."
    }
  ],
  "generation_target": "...",
  "metadata": {...}
}
```

### Top-k + Other Bucket KL

For teacher logits `z_T`, student logits `z_Q`, and teacher top-k set `K`:

```text
p_T(i) = softmax(z_T)_i
p_Q(i) = softmax(z_Q)_i
p_T(other) = 1 - sum_{i in K} p_T(i)
p_Q(other) = 1 - sum_{i in K} p_Q(i)
```

Loss:

```text
L_topk_other =
  sum_{i in K} p_T(i) [log p_T(i) - log p_Q(i)]
  + p_T(other) [log p_T(other) - log p_Q(other)]
```

This prevents unpenalized probability mass dumping outside the teacher top-k.

## Native-Context Training Mechanics

Full 1M-token backprop through the entire prefix is not the first implementation target. Use a prefix-cache / detached-prefix strategy.

### Prefix-Detached Long-Context Distillation

For a long context:

1. Run the prefix up to a selected training window to build KV/cache state.
2. Detach the prefix state.
3. Backprop through the selected supervised window and output span.
4. Compare student logits to teacher top-k+other labels at selected positions.

This trains the model under native-context positional/KV conditions without storing a 1M-token activation graph.

The limitation is that gradients do not flow through the entire 1M prefix. That is acceptable for this phase because the goal is native-context survival and long-position calibration, not full-sequence reconstruction through every prefix token.

### Output-Side 32k Distillation

For long outputs, use teacher-forced assistant tokens first:

```text
input prefix: up to 1M tokens
assistant output: up to 32k tokens
supervision: dense over output tokens
loss: top-k+other KL + format weighting
```

Then add on-policy generation:

```text
student generates 32k tokens
detect first collapse/failure
teacher labels student prefix around failure
train dense failure windows
```

On-policy is not useful while generation is total punctuation collapse. It becomes decisive once the model produces some meaningful tokens.

## Training Objective

Base loss:

```text
L = L_KD + lambda_code L_code + lambda_fmt L_format + lambda_on L_on_policy
```

### Distillation Loss

```text
L_KD = E_{positions} KL_topk_other(p_T(. | x), p_Q(. | x))
```

Use every position for short/medium records. Use selected windows for 1M records.

### Codebook Loss

For master weights `u` and hard quantized weights `Q(u)`:

```text
L_code = E[(u - Q(u))^2] / (E[u^2] + eps)
```

Weight:

```text
lambda_code = 0.001 initially
```

Do not over-weight early. The main goal is behavior, while hard-forward STE already enforces codebook behavior in the forward pass.

### Format Loss

Protocol-sensitive tokens:

```text
<think>
</think>
<predicted_observation>
</predicted_observation>
newline
brackets
quotes
EOS / stop
```

Weight:

```text
w_t = 1 + lambda_fmt * sum_{z in F} p_T(z | x_t)
```

Use only after the model is not in total punctuation collapse, or use lightly.

### On-Policy Loss

Student generates prefixes:

```text
x_Q ~ P_Q
```

Teacher labels those exact prefixes:

```text
L_on = KL_topk_other(p_T(. | x_Q), p_Q(. | x_Q))
```

This addresses exposure bias and loop/punctuation attractors.

## Optimizer

### Current One-A100 Feasible Optimizer

`factored_adafactor`:

- factored row/column second moment for 2D tensors,
- unfactored second moment for small 1D tensors,
- no first moment,
- supports hard-forward all-2D QAT on one A100.

Current fit evidence:

```text
Qwen3.5-9B all-2D ternary
BF16 masters
all trainable 2D + norms
context <= 1024
fits on A100 80GB
```

### Serious Multi-GPU Optimizer

For the real native-context run:

- 8-bit AdamW or Adafactor,
- FSDP/ZeRO sharding for master weights and optimizer state,
- cosine LR decay,
- LR higher than FP fine-tune: start with `1e-5` to `5e-5` sweeps.

Avoid alpha ramp. The forward path should be hard quantized from step 0:

```text
W_fwd = Q(u) + (u - stopgrad(u))
```

For Run A, group scales are recomputed from the current master weights on every
forward using the quantizer's group statistic. They are not slow learned FP16
parameters. If scale learning is introduced later, it must be a separate arm
because stale or slow scales can independently create mass/top1 plateaus.

## Experiment Stages

### Stage 0: BF16 Native-Context Baseline

Run:

```text
BF16 Qwen3.5-9B
1M input
32k output
fixed text/code/image/video scenarios
AgentWorld fixed10 long-form variants
```

Outputs:

- pass/fail for native context,
- throughput and memory,
- 1M prefill wall-clock,
- 32k generation wall-clock,
- peak VRAM and KV-cache footprint,
- effective tokens/second for prefill and decode,
- output validity,
- exact stop behavior,
- prompt/template compatibility.

Do not proceed until BF16 passes.

### Stage 1: Hard-Forward Dense KD Slope

Purpose: estimate whether hard-forward all-2D ternary learns at all with dense supervision.

Run:

```text
context: 1k-8k
data: streaming text/reasoning/multimodal
labels: every position top-64/128 + other
steps: enough for 100M-1B supervised tokens
optimizer: factored_adafactor on one A100, or sharded 8-bit Adam/Adafactor on multi-GPU
```

Required metrics:

- PTQ hard projection baseline on the same held-out source shards,
- held-out dense cache KL,
- held-out top1,
- student mass on teacher top-k,
- generation smoke,
- codebook forensics.

Held-out data must be split by source/dataset family, not only by row. Row-level
splits can contain near duplicates and will overstate generalization.

Pass signal:

```text
held-out KL decreases
held-out top1 increases
generation changes from punctuation collapse to alphanumeric/coherent
code_changed_frac and scale movement become nontrivial
```

Quantitative slope gate:

```text
fit held-out KL(t) = a * t^(-b) + c
project tokens-to-threshold for KL, top1, and generation smoke
continue only if projected cost is inside the planned token budget
```

Do not accept an eyeballed training-stream decrease as a positive slope.

Fail signal:

```text
training KL improves only on stream rows
held-out/generation unchanged
codebook forensics static
```

### Stage 2: Native-Context Teacher-Forced Distillation

Purpose: expose the compressed model to real long positions and 1M-prefix KV conditions.

Run:

```text
input: 64k -> 256k -> 1M curriculum
output: 1k -> 8k -> 32k teacher-forced assistant spans
labels: dense output tokens + selected input anchors/windows
```

Training uses detached long-prefix cache and backprops through selected windows/output spans.

Pass signal:

- long-position KL improves,
- BF16/control and compressed outputs share format,
- no sudden collapse when prompt length increases.

### Stage 3: Native On-Policy Survival

Purpose: eliminate long-output collapse.

Loop:

1. Student generates up to 32k tokens on fixed long tasks.
2. Detect collapse:
   - punctuation collapse,
   - newline collapse,
   - `<think>` loops,
   - missing `<predicted_observation>`,
   - premature EOS,
   - gibberish.
3. Teacher labels windows around the failure prefix.
4. Train on those windows mixed with bulk KD.

Pass signal:

```text
fixed long scenarios produce valid 32k-capable outputs
collapse rate decreases across rounds
valid observation / answer markers appear
```

### Stage 4: Pack And Evaluate

Pack:

```text
Q2_0-style ternary g128 codes
FP16 scales
continuous 1D norms
```

Evaluate:

- short perplexity/KL,
- medium instruction following,
- AgentWorld fixed10,
- AgentWorld 100-row,
- 1M/32k native-context suite,
- multimodal image/video suite,
- exact byte size and compression ratio.

## Metrics

### Training Metrics

`KL_topk_other`

- Distillation mismatch over teacher top-k plus tail bucket.
- Lower is better.

`top1`

- Fraction of positions where student argmax equals teacher argmax.
- Sensitive to head/logit ordering.

`student_top_mass`

- Student probability mass assigned to teacher top-k tokens.
- Early recovery often shows up here before top1.

`teacher_top_mass`

- How much teacher probability is captured by top-k.
- If low, top-k is too small for that row.

`format_mass`

- Teacher probability on protocol tokens.
- Used to identify format-critical positions.

### Generation Metrics

`valid_observation`

- AgentWorld-style valid `<predicted_observation>` extraction.

`parseable`

- Output can be parsed by the harness, even if semantically wrong.

`failure_kind`

- newline collapse,
- punctuation collapse,
- think loop,
- premature stop,
- gibberish,
- empty output.

`alnum_fraction`

- Cheap signal for punctuation collapse.

`generated_tokens`

- Whether the model can sustain long output without early EOS.

### Native-Context Metrics

`prefill_success`

- 1M-token input can be processed.

`long_position_KL`

- KL at selected positions near the end of long prefixes.

`answer_span_KL`

- Dense KL over long assistant output spans.

`long_generation_survival`

- 32k generation completes without collapse.

`retrieval_accuracy`

- Correct use of facts from far-back context.

### Codebook Forensics

`code_changed_frac`

- Fraction of ternary codes that changed relative to PTQ/base projection.

`active_sign_flip_frac`

- Nonzero sign flips.

`support_added_frac / support_dropped_frac`

- Movement into/out of zero support.

`scale_ratio_mean/std`

- Group scale movement relative to PTQ/base projection.

`current_zero_frac`

- Final ternary sparsity.

Expected healthy movement:

```text
nontrivial code changes
module-specific scale changes
some support movement
eventual generation improvement
```

Static forensics plus bad generation means the model is not becoming codebook-native.

## First Native-Context Test Suite

Use 20 fixed scenarios:

1. 1M text needle retrieval, short answer.
2. 1M text multi-needle synthesis, 2k answer.
3. 1M legal/report QA, citation answer.
4. 1M multilingual retrieval, cross-lingual answer.
5. 1M codebase navigation, patch explanation.
6. 1M codebase bug localization.
7. 512k text + 512k logs, incident summary.
8. 1M chat history, preference-consistent answer.
9. 1M math scratch corpus, final theorem/proof selection.
10. 1M mixed distractor retrieval.
11. 256k prompt, 32k coherent essay.
12. 512k prompt, 32k structured report.
13. 1M prompt, 32k structured report.
14. image document QA with long textual context.
15. chart/table image set with long context.
16. screenshot workflow trace with long text.
17. video temporal ordering QA.
18. multi-video comparison.
19. AgentWorld fixed long Android scenario.
20. AgentWorld fixed long terminal/MCP scenario.

For every scenario, run:

- BF16 teacher,
- current compressed candidate,
- previous best compressed candidate if available.

Do not judge with paid LLMs until degeneracy gates pass.

## First Concrete Run Plan

### Build Native Cache Shards

Shard types:

```text
dense_8k_text_reasoning_top64_other
dense_8k_multimodal_top64_other
native_64k_windowed_top64_other
native_256k_windowed_top64_other
native_1m_windowed_top64_other
native_output_32k_top64_other
onpolicy_failure_windows_top64_other
```

Initial target:

```text
100M supervised positions across dense 1k-8k shards
~100 native 64k records with sparse/window labels
defer 256k and 1M records until Stage 1 passes
100 fixed native 32k-output records
```

Reason: labeling 1k records at 1M tokens is already roughly 1B teacher prefill
tokens before training begins. That spend is only justified after dense
hard-forward learning has a measured held-out slope.

### Run A0: Existing-Checkpoint FP16 Head/Embed Control

Before training more, evaluate the current hard checkpoint with the same patch
but leave embeddings and lm_head BF16:

```bash
python experiments/qwen_agentworld/run_long_generation_gate.py \
  --model-dir /workspace/models/Qwen3.5-9B \
  --data-dir /workspace/AgentWorldBench_round8_fixed10_20260629 \
  --weight-patch /workspace/agentworld_runs/qwen35_dense_kd_curve_20260702/dense_hard_adafactor_128_patch.safetensors \
  --patch-exclude-regex '(^model\.embed_tokens\.weight$|^lm_head\.weight$|embed_tokens\.weight$)' \
  --lengths 256 \
  --max-rows 10 \
  --max-model-len 8192 \
  --out-dir /workspace/agentworld_runs/native_qat/head_embed_fp16_control_128patch
```

If punctuation collapse remains unchanged, head/embed are not the only blocker.
If top1/generation jumps materially, stage lexical quantization later.

Initial result on the existing 128-step artifact:

```text
rows: 10
valid_observation: 0
parseable: 10
failure_counts:
  gibberish: 7
  punctuation_collapse: 3
min_generated_tokens: 90
max_generated_tokens: 256
avg_generated_tokens: 225.8
patch tensors skipped:
  lm_head.weight
  model.language_model.embed_tokens.weight
```

Interpretation: FP16 head/embed changes the failure texture from mostly pure
punctuation to repetitive alphanumeric/list loops on several rows, but it does
not rescue behavior. The 128-step artifact is therefore not merely head/embed
strangulation; the ternary body is still not behavior-preserving.

### Run A1: FP32-Master Integration Probe

This is a 200-step unit test for the update path, not a quality run.

```bash
python experiments/qwen_agentworld/train_all2d_ternary_qat.py \
  --model-dir /workspace/models/Qwen3.5-9B \
  --cache-glob "/workspace/agentworld_runs/native_cache/dense_8k_*.pt" \
  --cache-shard-steps 256 \
  --mode ternary \
  --group-size 128 \
  --master-dtype float32 \
  --train-layers all \
  --train-embed \
  --train-head \
  --train-norms \
  --max-train-tokens 8192 \
  --optimizer factored_adafactor \
  --lr 1e-4 \
  --adafactor-beta2 0.999 \
  --adafactor-update-chunk-rows 1024 \
  --grad-clip 0.5 \
  --max-update-rms 0.5 \
  --steps 200 \
  --alpha-start 1.0 \
  --alpha-end 1.0 \
  --topk-other-bucket \
  --format-boost 0.0 \
  --code-weight 0.001 \
  --master-delta-sample-size 2048 \
  --save-latest-every 0 \
  --out-patch /workspace/agentworld_runs/native_qat/integration_fp32_200_patch.safetensors \
  --out-report /workspace/agentworld_runs/native_qat/integration_fp32_200_report.json
```

Pass condition:

```text
sampled master_delta_mean_abs scales with LR * steps
master_delta_moved_frac is nonzero across module families
global grad clip scale is logged and not silently crushing every update
offline codebook forensics move above the prior BF16-master floor
```

If this does not pass, do not launch any longer run.

Initial implementation result:

```text
all-ternary FP32 masters:
  status: OOM on one A100
  first failure fixed by row-chunked Adafactor update
  remaining failure: forward materialization of huge quantized embed/head
  observed allocation: about 3.79 GiB with less than 1 GiB free

ternary body + BF16 embed/head, FP32 masters:
  status: 3-step smoke passes
  trainable modules: 248
  sampled master values: 63,488
  step 1 master_delta_mean_abs: 2.51e-5
  step 3 master_delta_mean_abs: 4.17e-5
  moved fraction: ~0.99998
  grad_clip_scale:
    step 1: 5.51e-4
    step 2: 3.44e-3
    step 3: 1.08e-2
```

Interpretation: FP32 body masters integrate correctly. Strict all-ternary
including embedding/head is not one-A100 feasible with the current forward
materialization path. The practical one-A100 slope arm is ternary body with
BF16 lexical boundary, or a future memory-optimized embedding/head quantized
forward. A true all-ternary FP32-master run likely needs multi-GPU sharding or
a custom chunked embedding/head path.

Follow-up one-A100 integration result:

```text
run:
  fp32_window_early_late_integration_200_20260702T034500Z
policy:
  full hard-ternary body forward
  BF16 embed/head
  FP32 trainable masters for layers 0,1,2,3,28,29,30,31 + norms
  dense top64+other KD, alpha=1 from step 0
  lr=1e-4, factored Adafactor, max_train_tokens=1024
status:
  completed 200/200 steps
  final patch: 3.3 GB
  trainable modules: 62
```

Selected metrics:

```text
step 1:
  kl: 7.7470
  top1: 0.0156
  student_top_mass: 0.0875
  grad_clip_scale: 6.27e-4
  master_delta_mean_abs: 2.27e-5
  master_delta_rms: 4.35e-5
  master_delta_moved_frac: 0.9999

step 100:
  kl: 4.6267
  top1: 0.4000
  student_top_mass: 0.3768
  grad_clip_scale: 8.45e-2
  master_delta_mean_abs: 2.07e-4
  master_delta_rms: 4.19e-4

step 200:
  kl: 5.4884
  top1: 0.0596
  student_top_mass: 0.2726
  teacher_top_mass: 0.9977
  grad_clip_scale: 0.1594
  master_delta_mean_abs: 2.86e-4
  master_delta_rms: 6.10e-4
  master_delta_rel_mean_abs: 0.0242
```

Interpretation: this is not yet a quality result, because only early/late
windows are trainable and the final patch is not a complete packed hard-ternary
artifact. It does prove the corrected update path: FP32 masters move
continuously, the global clip scale recovers after the first large updates, and
dense KD can reduce cache loss on seen rows under the hard-forward constraint.

### Run A2: LR Sweep

Run 2k-5k steps at:

```text
lr in {2e-5, 1e-4, 5e-4}
warmup: 500-1000 steps if implemented by scheduler
master dtype: FP32
alpha: 1 from step 0
```

Each arm must report:

- held-out-source KL/top1/top-k mass;
- generation smoke;
- codebook forensics;
- sampled master movement;
- grad clip scale.

The 100k-step dense arm only runs after choosing LR from this sweep.

Initial measured LR sweep on the feasible one-A100 window setup:

```text
shared setup:
  full hard-ternary body forward
  BF16 embed/head
  train layers 0,1,2,3,28,29,30,31 + norms
  max_train_tokens=1024
  report-only runs where possible, no multi-GB final patches

lr=2e-5:
  status: completed 200/200
  step 1:
    kl: 7.7470
    top1: 0.0156
    master_delta_mean_abs: 4.53e-6
  step 128:
    kl: 2.7605
    top1: 0.2422
    student_top_mass: 0.6949
    master_delta_mean_abs: 6.03e-5
  step 200:
    kl: 0.8487
    top1: 0.7275
    student_top_mass: 0.9583
    master_delta_mean_abs: 7.84e-5
    master_delta_rms: 1.25e-4
    master_delta_rel_mean_abs: 0.00664

lr=1e-4:
  status: completed 200/200
  step 200:
    kl: 5.4884 on the final OpenR1 row
    top1: 0.0596 on the final OpenR1 row
    master_delta_mean_abs: 2.86e-4
    master_delta_rms: 6.10e-4
    master_delta_rel_mean_abs: 0.0242
  note:
    stable, but more aggressive; row-by-row metrics are noisier than lr=2e-5
    because this was the first integration run and not a held-out slope fit.

lr=5e-4:
  status: stopped at step 129 as a determined failure
  step 1:
    master_delta_mean_abs: 1.13e-4
    master_delta_max_abs: 1.01e-2
  step 50:
    kl: 12.7578
    top1: 0.0173
    master_delta_mean_abs: 4.41e-4
    master_delta_max_abs: 5.43e-2
  step 128:
    kl: 8.5467
    top1: 0.0391
    master_delta_mean_abs: 4.97e-4
    master_delta_max_abs: 8.13e-2
  step 129 OpenR1:
    kl: 10.8161
    top1: 0.0049
    student_top_mass: 0.0141
```

Interpretation:

```text
5e-4 is too destructive for this optimizer/setup.
2e-5 is stable and produces the best short measured cache slope.
1e-4 is still viable and gives much larger code movement, but needs a matched
held-out-source slope run before it receives the long budget.
```

The next non-wasteful experiment is not native 1M cache construction. It is a
matched held-out-source dense slope run at `2e-5` versus `1e-4`, with generation
gates only after the hard cache metrics clear a non-degenerate threshold.

### Run A2.5: Chunked Head KD Kernel

Rationale:

```text
Naive lm_head fake-quant + full logits confounds the paired slope arms.
It materializes Q(u), logits, softmax/logsumexp intermediates, and sometimes
a full head gradient. That can change max tokens/step between all-ternary and
BF16-head controls.
```

Implemented changes:

```text
TrainQuantEmbedding now uses gather-then-quantize for trainable embeddings.
  This avoids full-table fake quantization in the embedding forward.

--chunked-head-kd bypasses model.lm_head in the HF forward.
  It calls model.model(...) to get final hidden states, then computes
  top-k+other KD through lm_head in vocab chunks.

ChunkedTopKOtherHeadKL:
  forward:
    stream vocab chunks
    recompute quantized head rows per chunk
    accumulate student logsumexp in FP32
    gather student logits for cached teacher top-k indices
    compute coarsened top-k+other KL
  backward:
    stream vocab chunks again
    use closed-form logit gradient
    propagate grad_hidden
    propagate STE grad_weight when the head is trainable
```

Current limitation:

```text
The first implementation still returns a full head gradient when lm_head is
trainable. It removes full-logit and full-Q(u) head materialization, but it is
not yet the final immediate-update/no-full-head-grad kernel.
```

Correctness result:

```text
CPU parity in the A100 Python 3.10 environment:
  mode=none:
    loss error: 2.98e-8
    hidden grad relative error: 1.94e-7
    weight grad relative error: 1.41e-7
  mode=ternary alpha=1:
    loss error: 0
    hidden grad relative error: 1.74e-7
    weight grad relative error: 1.13e-7
  mode=ternary alpha=0.35:
    loss error: 5.96e-8
    hidden grad relative error: 1.86e-7
    weight grad relative error: 1.62e-7
```

GPU smoke results:

```text
early/late trainable window + trainable ternary embed/head:
  status: passed 2 steps at 1024 max tokens
  trainable modules: 64
  chunked head KD active
  anti_weight: 0
  final observed:
    step 2 KL: 15.4415 on short Aya row
    master_delta_mean_abs: 6.13e-6

all layers trainable + trainable ternary embed/head:
  status: passed 1 step at 1024 max tokens
  trainable modules: 250
  step 1 KL: 8.4558
  master_delta_mean_abs: 4.88e-6

all layers trainable, 20-step attempt:
  status: failed on step 2 with CUDA OOM
  failure site: MLP TrainQuantLinear alpha_quantized_weight
  attempted allocation: 192 MB
  free memory at failure: about 135 MB
```

Interpretation:

```text
The old head/embed materialization blocker is removed at the 1k-token smoke
scale. The remaining one-A100 blocker is full all-layer FP32 masters/states plus
per-linear quantized-weight materialization in the trunk. This points to either:

  1. BF16 masters with stochastic rounding / Kahan compensation;
  2. chunked or custom-autograd trunk linear kernels;
  3. rotating-window all-layer coverage;
  4. multi-GPU sharding for the real all-2D run.
```

### Run A3: Paired Dense Hard Slope

Run the winning LR as two matched arms:

```text
arm 1: all-2D ternary, including embed_tokens and lm_head
arm 2: ternary body, BF16 embed_tokens, BF16 lm_head
```

For arm 2, use:

```bash
  --skip-quant-embed \
  --skip-quant-head
```

This isolates whether all-ternary lexical boundaries are suppressing top1. The
final artifact can still be strict ternary by quantizing the head/embed in a
later focused KD phase.

### Run B: Native Window Training

Needs a new trainer or extension:

```text
train_native_context_window_qat.py
```

Required features:

- load long record manifest,
- prefill long prefix with no gradient,
- reuse within-record prefix/KV state across multiple supervised windows,
- select supervised window,
- run student and teacher labels for window/output positions,
- backprop through selected window only,
- support 1M positional/KV conditions.

### Run C: 32k Output Survival

Needs:

```text
collect_native_onpolicy_survival_cache.py
train_native_onpolicy_survival_qat.py
```

Loop:

```text
student generate 32k
detect failure
cache teacher labels around failure
train on failure windows
repeat
```

## Stop/Continue Criteria

Continue if:

- held-out dense KL improves,
- held-out top1 improves,
- student top-k mass rises,
- generation failure shifts from punctuation to alphanumeric/coherent,
- codebook forensics move,
- native long-position KL improves.

Stop or relaunch if:

- training-stream KL improves but held-out/generation does not,
- generation remains punctuation collapse after meaningful token budget,
- codebook forensics remain static,
- BF16 native baseline itself fails,
- memory forces non-native shortcuts that invalidate the experiment.

## Interpretation Of The Latest 128-Step Result

The latest run is a bug report, not a baseline.

Observed:

```text
hard-forward all-2D ternary QAT fits with factored Adafactor
dense top-64+other cache works
training-stream rows can improve
free generation remains punctuation collapse
codebook changes are ~2.5e-5 globally
sign flips are near zero
scales are ~1.0x
master dtype was BF16
```

Conclusion:

```text
The model has not entered a codebook-native regime, and the update path may
have been quantized away by BF16 master ULPs.
```

For weights around `0.02`, BF16 ULP is on the order of `1e-4`. A `1e-5`
per-step update can round to no-op under nearest-even accumulation. The tiny
code-change count is therefore consistent with frozen masters, not merely with
too few steps.

The next meaningful test is not "run longer." It is:

```text
FP32/stochastic master update path
200-step integration proof
128-step rerun with healthy forensics
LR sweep
paired all-ternary vs BF16-head/embed slope
```

Only after that should supervised tokens scale by orders of magnitude.

## Deliverables

1. BF16 native-context baseline report.
2. Native cache manifest format and builder.
3. Dense 8k top-k+other teacher cache shards.
4. Native 64k/256k/1M sparse-window teacher cache shards.
5. Hard-forward all-2D QAT slope report.
6. Native 1M/32k generation gate.
7. Codebook forensics report every checkpoint.
8. Final packed Q2_0-style artifact if behavior passes.

## Recommendation

Run this as a two-track effort:

1. One A100 continues infrastructure and slope probes.
2. Multi-GPU rental runs the real dense-token and native-context program.

The decisive investment question is not whether the 128-step artifact is good.
It is not, and it is not valid slope evidence until the master-update path is
proven. The question is whether FP32-master hard-forward dense KD shows a
positive held-out-source slope and healthy codebook movement before the first
1B-2B supervised tokens.

Launch order:

1. FP16 head/embed eval of the existing checkpoint.
2. FP32-master integration test.
3. 128-step forensics rerun.
4. LR sweep.
5. Paired Run A arms.
6. Dense-token scale-up.
7. Native 64k cache.
8. Native 256k/1M cache only after Stage 1 passes.

If the paired dense slope is positive, scale to 10B+ tokens and then run native
survival. If no arm shows positive held-out-source slope, relax the artifact to
mixed precision around embeddings/head/early-late blocks and target 6.5x-7x
instead of strict all-2D ternary.
