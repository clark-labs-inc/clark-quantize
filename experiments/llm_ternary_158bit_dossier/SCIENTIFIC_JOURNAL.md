# LLM 1.58-bit Ternary Research Journal

Status date: 2026-07-30

## Scope and record policy

This dossier is the navigation and scientific interpretation layer for Clark's
LLM ternary-conversion work. It does not replace, rename, or move the historical
run records under `experiments/qwen_agentworld/`, the root-level Qwen reports,
or the frozen large-model protocol under `experiments/large_scale_ternary/`.
Those linked artifacts remain authoritative for commands, metrics, and
checkpoint provenance.

The large-model program is inactive. BackdropKit's Swin2SR work is the sole
active optimization target; LLM results are retained only as scientific
reference.

## Verified findings

### 1. Continuous masters and hard low-bit forwards are the durable invariant

Across the repository, the most defensible transferable mechanism is:

```text
continuous master weights
  -> hard ternary forward
  -> straight-through gradients
  -> code changes when masters cross quantization boundaries
```

The optimizer audit explicitly concludes that this representation, rather than
any branded optimizer, is Clark's strongest supported contribution. Optimizer
identity must be established by matched experiments rather than inferred from
run names or endpoint weight geometry.

Evidence:

- [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md)
- [Qwen3.5 ternary trainer recipe](../../TERNARY_QWEN35_RECIPE.md)
- [hard-ternary 4B run report](../../QWEN35_4B_GLOBAL_TERNARY_RUN_20260711.md)

### 2. The strongest recorded Qwen3.5-4B run improved teacher-behavior metrics

The `qwen35_4b_NORMCOMP_fullKL_anchorDecay_5B_20260711` run used hard ternary
group-128 forwards, continuous masters, `muon-fast` for matrix masters,
continuous trainable norms, and a temporary hidden-state anchor. Its final
logged probe at step 19,800 reported:

- mean full KL `0.4246758`;
- top-1 agreement `75.1248%`;
- student top-128 mass `93.3082%`;
- code churn `4.4472%`;
- sign inheritance `99.8896%`;
- zero fraction `31.6214%`.

Against the earlier full-KL control reported in the same run record, mean KL
was 27.6% lower and top-1 agreement was 3.98 percentage points higher. Behavior
continued improving after the hidden anchor reached zero, supporting temporary
residual-stream guidance and continuous norm compensation as useful mechanisms.

This is not a lossless-quality result. The final probe is later than the latest
durable step-19,000 checkpoint, and no post-stop generation or downstream
benchmark suite was run.

Evidence: [Qwen3.5-4B global run](../../QWEN35_4B_GLOBAL_TERNARY_RUN_20260711.md)

### 3. Code homotopy is sharply layer-specific

The Qwen-AgentWorld record establishes that boundary-near candidates can move
codes while bottom-absolute candidates are mostly inert. Soft-forward warmup
rescued some individual layers, but:

- aggressive broad late-MLP movement collapsed;
- layers that passed individually did not compose safely by post-hoc stacking;
- a fixed-10 canary could conceal repetition failures that appeared in a
  larger fixed-100 screen.

Movement fraction alone is therefore not a promotion metric. Joint training
and a larger behavior gate are required before combining individually safe
surfaces.

Evidence: [Qwen AgentWorld status](../qwen_agentworld/STATUS.md)

### 4. Cache-level quality does not prove agent behavior

The strict 8x Qwen-AgentWorld policy produced a live, non-collapsed token
stream and competitive cache metrics, yet its official seven-domain smoke score
was `0.00`; outputs were malformed rather than useful task actions. Conversely,
the small ternary-only MoE smoke scored better than its BF16 smoke reference,
but the seven-row, 64-token test is only a canary.

Any LLM promotion gate must therefore include task-format and system-behavior
evaluation, not only KL, top-1 agreement, or generation non-collapse.

Evidence:

- [Qwen AgentWorld status](../qwen_agentworld/STATUS.md)
- [Qwen AgentWorld protocol README](../qwen_agentworld/README.md)

### 5. Full-fidelity AgentWorld evaluation has a hard infrastructure boundary

The official Qwen-AgentWorld boundary uses 262,144 context, 32,768 generation
tokens, and tensor parallelism 4. The recorded single-A100 work is explicitly
reduced-context smoke evidence. The full-fidelity preflight correctly refused
to run with one GPU.

Evidence:

- [Qwen AgentWorld protocol README](../qwen_agentworld/README.md)
- [Qwen AgentWorld status](../qwen_agentworld/STATUS.md)

### 6. Packing and quality are separate research axes

Clark's pack40 representation is near the ternary entropy floor and can be
smaller than aligned Q2_0 at identical codes. That does not establish that the
codes retain model quality. Training quality, packed numerical equivalence, and
runtime performance require separate gates.

Evidence:

- [Prism-level compression analysis](../../PRISM_LEVEL_COMPRESSION.md)
- [Bonsai ternary forensics](../../bonsai_ternary_deep_forensics.md)
- [archived large-model research program](../large_scale_ternary/RESEARCH_PROGRAM.md)

## Stable negative results

- A global Qwen3.5-9B mirror-SGD ramp reached the codebook while behavior was
  already collapsed; codebook proximity is not behavioral recovery.
- Full all-module codebook losses and unbounded optimizer intermediates caused
  A100 OOMs. Auxiliary losses, update statistics, checkpoint writes, and prox
  operations must be memory-bounded.
- Aggressive local code-homotopy movement collapsed several layers; individually
  safe patches failed when naively stacked.
- A strict 8x policy with acceptable cache/logit measurements scored zero on
  the tiny AgentWorld system smoke.
- Forensic endpoint signatures do not identify a proprietary optimizer and
  must not be optimized as substitutes for behavior.

Evidence:

- [global mirror 10k record](../qwen_agentworld/GLOBAL_T2_MIRROR_10K_RUN.md)
- [global fulltrain v1](../qwen_agentworld/GLOBAL_T2_FULLTRAIN_V1_RUN.md)
- [global fulltrain v2](../qwen_agentworld/GLOBAL_T2_FULLTRAIN_V2_RUN.md)
- [Qwen AgentWorld status](../qwen_agentworld/STATUS.md)
- [optimizer audit](../large_scale_ternary/OPTIMIZER_AUDIT.md)

## Unresolved questions

1. Does the durable step-19,000 Qwen3.5-4B checkpoint reproduce the later
   step-19,800 probe trajectory when evaluated directly?
2. Does it materially beat PTQ ternary and a hardware-relevant int4 control on
   a sealed multilingual and capability suite?
3. Which gains came from trainable norms, temporary hidden anchors,
   `muon-fast`, the full-KL objective, or their interaction?
4. Can a packed graph reproduce unpacked logits within a registered tolerance
   and provide real CPU/Apple throughput and memory benefits?
5. Can safe code homotopy be made jointly trainable across layers rather than
   composed from independently trained patches?

## Next gates if this inactive lane is ever resumed

These gates require a fresh explicit user decision before model loading or
compute:

1. Recover the step-19,000 checkpoint and verify SHA-256
   `47c1e132d1ba92975e25d1211f883b496f009b2ca50b38dd0009853e352a5822`.
2. Evaluate the exact checkpoint, exact BF16 parent, PTQ ternary control, and
   hardware-relevant int4 control with the sealed Gate B protocol.
3. Run the equal-language likelihood suite and capability/system-behavior
   gates; do not promote from cache metrics alone.
4. Pack the exact checkpoint and prove unpacked-logit equivalence.
5. Measure packed bytes, peak memory, throughput, and energy on Apple Silicon
   and x86 CPU.

Evidence and protocol: [archived large-model research program](../large_scale_ternary/RESEARCH_PROGRAM.md)

## Journal maintenance

Append dated entries only when a new artifact or reproducible receipt changes a
finding. Each entry should name the exact model revision, source revision,
dataset manifest, checkpoint hash, controls, and whether the evidence is a
smoke, development result, or sealed holdout.

## 2026-08-01 — Status review

No LLM model is being loaded, evaluated, or trained under the BackdropKit
program. The Qwen and large-scale ternary material remains archived as
scientific reference for optimizer, schedule, packing, and evaluation design.
Its unresolved gates and evidence boundaries above are unchanged.
