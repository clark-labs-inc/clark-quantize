# Archived Clark large-model scaling program

Status date: 2026-07-29

> **Inactive reference only.** BackdropKit's shipped tiny browser models are
> the sole active optimization target. This large-model program, its model
> selection gates, and its language-manifest jobs are frozen. Existing
> artifacts are retained for provenance, but no compute or research decisions
> should be spent here. Reusable quantizers, optimizer lessons, packing, and
> evaluation methods may inform the active Swin2SR work in
> `experiments/browser_swin2sr/`.

## Completion standard

The program is complete only when all of the following are true for a model
with at least one billion parameters:

1. A hard-ternary checkpoint exists with a content hash and reproducible
   provenance.
2. Matmul-heavy weights use `{-1, 0, +1}` codes. Every higher-precision island,
   scale, embedding, activation, and cache is included in the effective-bit
   ledger; "1.58-bit" never means the idealized trit entropy alone.
3. A preregistered held-out suite compares the checkpoint with the exact
   full-precision parent and a hardware-relevant 4-bit control. It includes
   perplexity/KL, capability tasks, multilingual slices, and long-context
   behavior. Judge-model scores are supplementary, not the primary evidence.
4. Real packed bytes, peak memory, prompt/decode throughput, and energy are
   measured on at least Apple Silicon and x86 CPU. GPU and browser results are
   additional deployment lanes, not substitutes for those two.
5. Training and export can be replayed from manifests, pinned source revisions,
   structured logs, and checkpointed optimizer/RNG/data-order state.

This definition deliberately separates a large checkpoint from a validated
large checkpoint. The current repository has the former, not yet the latter.

## Current evidence

| Lane | Strongest artifact | What is established | Missing proof |
| --- | --- | --- | --- |
| Archived dense LLM conversion QAT | Existing 4B checkpoint metadata | Historical logs report improving teacher KL, but this lane is dormant and was not exercised in the current program | The durable checkpoint and its parent would require explicit model approval before any recovery or evaluation |
| DiT QAT | Clark Air Sana 1.6B packed artifact | Hard low-bit execution and compact packing work at billion-parameter scale | Existing visual evidence is narrow; the mixed-rate model is about 1.85 effective bits, not a globally ternary quality-parity proof |
| Browser vision QAT | Swin2SR lightweight | A 1.6-bit trit payload, numerically verified WebGPU/WASM primitives, 206,500-image corpus, exact-resume QAT, and MSE-optimal/absmean/ParetoQ-SEQ controls exist | Full runtime graph and broad held-out evaluation are incomplete; the active 26k-step MSE-optimal run is now in fully hard recovery |
| Native ternary reference | Microsoft BitNet b1.58 2B4T | Native ternary training can work at 2B scale when trained from scratch on 4T tokens | This is an external reference, not a Clark-produced checkpoint |

The archived LLM and Sana artifacts are promising, but no current result justifies a
"without quality loss" claim.

## Research synthesis

### 1. Native training is the quality ceiling

[BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) is the strongest public
reference for native scale: ternary `BitLinear`, per-token int8 activations,
sub-layer normalization, a high-learning-rate bulk phase, and a lower-rate
high-quality cooldown over four trillion tokens. It shows that native training
can be competitive; it does not show that inexpensive conversion of an
arbitrary pretrained model is lossless.

### 2. Conversion below two bits is representation reconstruction

[ParetoQ](https://arxiv.org/abs/2502.02631) measures a transition below three
bits: ternary and two-bit QAT move much farther from the pretrained
distribution than three- or four-bit tuning. Its symmetric equalized
quantizer is therefore a required control against Clark's current
optimal-MSE and absmean quantizers. Short LoRA-like recovery is not an
adequate scaling assumption.

### 3. PTQ is a control and initializer, not the only path

[TWLA](https://arxiv.org/abs/2606.13054) combines ternary-friendly orthogonal
shaping, calibration-metric relocation, and inter-layer activation-bit
allocation. It makes a modern W1.58A4 PTQ control valuable, especially for
activation outliers. A PTQ result must still be evaluated separately from a
native or QAT result.

### 4. Vision needs explicit residual information tests

[BiMaCoSR](https://arxiv.org/abs/2502.00333) reports that sparse full-precision
and low-rank residual branches prevent binary super-resolution collapse.
[RobuQ](https://arxiv.org/abs/2509.23582) identifies activation quantization as
a DiT bottleneck and uses Hadamard shaping plus activation-only mixed
precision. Clark should ablate residual branches, rotations, and continuous
islands rather than hiding them outside the bit ledger.

### 5. Runtime format and mathematical entropy are different

[bitnet.cpp](https://github.com/microsoft/BitNet) proves optimized ternary CPU
and GPU inference, but Clark's target must report its exact packing. A
base-3 pack is close to 1.6 payload bits per weight; common aligned 2-bit
formats trade larger files for simpler kernels. Both belong on the frontier.

Prism's public Bonsai whitepapers document deployed formats and benchmarks but
do not disclose a reproducible conversion-training recipe. Clark's
weight-forensics are hypotheses to test, not ground truth about proprietary
training.

## Experimental ladder

### Gate A: trustworthy small-model harness

- Resume Swin2SR from step 4,000 with atomic, signal-safe checkpoints.
- Add held-out real UI, multilingual text, and natural-image strata.
- Compare optimal-MSE ternary, absmean, ParetoQ-style SEQ, and a rotation
  control under identical examples and optimizer budgets.
- Report mean, worst-decile, OCR-oriented edge quality, packed bytes, latency,
  and peak memory.

Exit criterion: one hard-ternary treatment has stable improvement across all
three strata and survives a fresh-seed replication.

Current implementation receipts:

- 206,500 training images and 3,040 held-out images have hashed manifests;
- exact resume, RNG/data-order replay, atomic save, and interruption behavior
  have automated tests;
- MSE-optimal, absmean, and the authors' exact row-wise ParetoQ SEQ treatment
  are implemented and test-pinned;
- the active MSE-optimal run reached its first fully hard checkpoint at step
  7,500 and continues into recovery training;
- on a preregistered-seed 200-sample held-out pilot, step 7,500 beat the prior
  3,000-step hard reference by `+0.9399 dB` ground-truth PSNR (paired bootstrap
  95% CI `[+0.7686,+1.1226]`), `+0.8235 dB` edge PSNR, and `+2.1708 dB`
  teacher PSNR; it won 87.0%, 87.5%, and 85.0% of paired samples respectively;
- the pilot contained 136 FLORES devtest canvases and 64 held-out RICO
  screenshots. Per-script cells are too sparse for language claims, so the
  full 3,040-sample evaluation remains mandatory after recovery;
- at step 8,500, the same pilot improved another `+0.1176 dB` over step 7,500
  (95% CI `[+0.0725,+0.1631]`) and reached `+1.0575 dB` over the old reference,
  showing early hard-recovery progress rather than collapse;
- exact export accounting shows the prior selected trunk at 1.86667 effective
  bits/weight after FP16 scales, but the protected-island model at 6.28301
  data bits/parameter; neither is mislabeled as a globally 1.58-bit model;
- all registered quantizer/shaping controls are now executable. The
  function-preserving block-Hadamard arm has unit, checkpoint, export, and full
  packed-artifact reconstruction proofs; matched-budget training remains to be
  run after the current arm finishes.

### Gate B: large-model audit after an explicit model decision

- Do not assume the archived 4B model is the next target.
- After the user explicitly selects a model/provider, recover its checkpoint
  and verify its hash against the archived manifest.
- Evaluate that checkpoint, its exact BF16 parent, PTQ ternary, and a 4-bit
  control with the same prompts, tokenizer, context, and harness revision.
- Measure per-arm KL/perplexity before any generative benchmark.
- Pack the exact checkpoint and verify unpacked logits against the training
  representation before benchmarking native inference.

Exit criterion: the QAT checkpoint materially beats PTQ at the same ternary
rate, has no catastrophic capability slice, and its packed runtime is
numerically equivalent within a registered tolerance.

Artifact recovery receipt, 2026-07-29: no local copy was found. A read-only
connection to the former `a100` alias stopped on a changed SSH host key; that
identity must be verified before the known-host entry is changed. The `4a100`
alias refused its configured connection. The expected checkpoint SHA-256 is
`47c1e132d1ba92975e25d1211f883b496f009b2ca50b38dd0009853e352a5822`.
No evaluation has been launched.

The Gate B capability protocol is now executable but remains unused. It:

- requires an explicit model ID, immutable model revision, local-Hugging-Face
  arm, sealed sample manifest, and output path;
- aborts if any checkpoint tensor is missing, duplicated, unmatched, or has a
  different shape instead of silently retaining a base-model value;
- freezes exact GSM8K and subject-balanced MMLU rows, MMLU support examples,
  dataset revisions/fingerprints, row hashes, and prompt-contract hash;
- records checkpoint, manifest, model-config, tokenizer-vocabulary, source,
  environment, per-sample prediction, timing, and confidence-interval receipts;
- provides a paired bootstrap comparison that rejects arms with different
  samples, model revisions, or task sets.

The frozen protocol is
`experiments/large_scale_ternary/gate_b_preregistration.json`. The likelihood,
PTQ-ternary, int4, energy, and full native packed-runtime lanes are explicitly
registered. The likelihood evaluator is now implemented; the other four lanes
remain incomplete, so the harness refuses to mislabel them. No LLM was loaded
or called while building this boundary.

The likelihood source is FineWeb2 release `v2.1.0`, pinned to commit
`66d1a3b989bad0c42b88179a7fb60c47ce87af74`. Repository metadata proves 1,318
language-script configurations have dedicated test shards; their ordered
universe is sealed at
`experiments/large_scale_ternary/fineweb2_v210_test_universe.json` with SHA-256
`d0a21fadf5919b5eb3153ae6c668d25234228d23529c063a45c16e0b33ae1a79`.
The active manifest job samples exactly one document from every available
configuration, records row hashes, and supports exact resume. A four-document
attempt was retained as a failed arm after the fifth configuration exposed a
one-document low-resource test split; reducing to one preserves equal coverage
instead of excluding the smallest languages. The evaluator reports token-micro,
equal-language macro, per-language, and worst-decile perplexity plus exact
full-vocabulary KL to the pinned BF16 parent.

### Gate C: controlled scaling run

- Run a factorial pilot over quantizer, optimizer, continuous compensation,
  rotation, and schedule on a sub-billion proxy.
- Promote only effects that replicate and improve behavior per packed bit.
- Apply the winning treatment to the 1B-4B lane with frozen manifests,
  periodic held-out gates, and stop rules based on quality plateaus rather
  than forensic signature matching.

Exit criterion: a hashed billion-parameter checkpoint passes the completion
standard above.

### Gate D: portable inference

- CPU: compare Clark base-3 packing with a bitnet.cpp-compatible aligned
  format.
- Apple: implement or adapt Metal/MLX kernels and measure unified-memory use.
- Web: keep a WebGPU kernel plus WASM SIMD fallback, but only claim browser
  support for graphs actually executed end to end.
- Verify identical decoded codes/scales and bounded output error across every
  backend.

## Preregistered ablations

| Factor | Levels |
| --- | --- |
| Initialization | absmean; optimal-MSE; activation-metric relocation |
| Ternary function | BitNet absmean; ParetoQ-style SEQ |
| Schedule | hard from start; attention then MLP; progressive global |
| Optimizer | AdamW control; BonsaiQAT8; Muon geometry |
| Compensation | none; trainable 1D norms; input redistribution; sparse/low-rank residual |
| Shaping | none; shared Hadamard; learned Kronecker rotation |
| Activations | FP16; int8; mixed 4-bit; mixed 2-4-bit where stable |

Each ablation must share seeds, samples, token/image exposures, validation
frequency, and checkpoint policy. Quality is reported against both ground
truth and the exact teacher. Failed and interrupted runs remain in the ledger.

## Archived next decision

Finish the active Swin2SR AdamW control, run its full held-out evaluation, then
execute the matched optimizer ladder in
`experiments/large_scale_ternary/OPTIMIZER_AUDIT.md`. This isolates Clark's
actual optimizer contribution on a small, inexpensive model before selecting
or spending compute elsewhere. Per the current product goal, the large-model
promotion described below is superseded: BackdropKit Swin2SR remains the only
active model until it passes full held-out quality and browser-runtime gates.
