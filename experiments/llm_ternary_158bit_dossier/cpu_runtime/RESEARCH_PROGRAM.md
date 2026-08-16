# CPU-First Ternary LLM Research Program

Status: active direction
Start date: 2026-08-16
Primary model: `Qwen/Qwen3.8-27B`, text-only runtime first
Development model: `Qwen/Qwen3.5-4B`

## North star

Produce a hard-ternary Qwen-family model whose primary deployment target is a
CPU process with no Python, PyTorch, CUDA, or Triton dependency. Training may
use GPUs; deployment quality is judged on CPUs.

The program is successful only when the packed model runs end to end on both
ARM64 and x86-64, retains measured capability against its exact BF16 parent,
and demonstrates a useful advantage over a strong hardware-relevant int4
control in at least one of memory, decode throughput, or energy.

This changes the order of work. The runtime contract is fixed before another
large QAT run. Training must learn the exact scales, activation rounding,
continuous islands, and tensor layouts that the CPU kernels execute.

## Decisions already made

1. **Two representations, one set of codes.** The canonical artifact uses
   exact radix-5 packing at 1.6 payload bits per trit. A machine-local compute
   image may transcode the same codes into an ISA-specific 2-bit or LUT-tiled
   layout. Canonical packing is optimized for distribution; the compute image
   is optimized for inner-loop execution.
2. **Text runtime first.** The vision tower and MTP are optional sidecars and
   are not loaded by the first CPU runtime. Their bytes are reported whenever
   a full multimodal artifact is claimed.
3. **Decode and prefill are separate products.** Decode is principally a
   matrix-vector and memory-bandwidth problem. Prefill is a matrix-matrix,
   attention, and chunked-recurrence problem. Each has its own kernels and
   scorecard.
4. **W1.6A16 is the correctness rung; W1.6A8 is the performance target.** A
   BF16/FP16-activation path establishes graph and weight parity. Per-token or
   per-tile int8 activation quantization is promoted only after a paired
   quality gate.
5. **GPU QAT serves the CPU runtime.** Global QAT remains a GPU workload. Pure
   CPU QAT is a portability experiment, not the planned way to spend the main
   token budget.
6. **No new 27B bulk run before the 4B runtime gate.** The existing 4B success
   is the development checkpoint. It must first execute through the real
   packed CPU graph, including the hybrid recurrent blocks.

## Exact Qwen3.8-27B text graph

The pinned model config has 64 text layers with a 3:1 hybrid pattern: 48 Gated
DeltaNet layers and 16 full-attention layers. The principal dimensions are:

| Surface | Shape or count |
| --- | ---: |
| Hidden width | 5,120 |
| MLP width | 17,408 |
| Full-attention query heads | 24 x 256 |
| Full-attention KV heads | 4 x 256 |
| DeltaNet key heads | 16 x 128 |
| DeltaNet value heads | 48 x 128 |
| DeltaNet convolution | depthwise, width 4, 10,240 channels |
| Vocabulary | 248,320 |

The full-attention query projection emits both query and output-gate channels,
so its output width is 12,288. Each DeltaNet block projects `qkv` to 10,240,
`z` to 6,144, scalar gates `a` and `b` to 48 each, then projects the recurrent
output from 6,144 back to 5,120.

The recurrent state is not a KV cache. At FP32 it is about 3 MiB per DeltaNet
layer, or 144 MiB for all 48 layers at batch one. The full-attention KV cache
is about 64 KiB per generated token in BF16 across all 16 attention layers:
about 2 GiB at 32K context or 16 GiB at 256K. An int8 KV lane halves those
figures and therefore belongs in the core runtime plan.

The untied embedding and LM head are each about 1.27B parameters. Leaving both
in BF16 costs roughly 4.74 GiB total and would dominate the continuous-island
ledger. The CPU product must explicitly evaluate Q6/Q8 embeddings and a
ternary, Q4, or Q6 head; these tensors cannot remain an unreported exception.

## Artifact and runtime architecture

```text
canonical .cq3 artifact
  header + tokenizer/config hashes
  tensor directory and shapes
  radix-5 ternary codes
  FP16 group-128 scales
  continuous-island tensors and their exact dtypes
  per-section checksums
          |
          | one-time verified transcode
          v
ISA-specific sidecar compute image
  AVX2 / AVX-512-VNNI / ARM64-DOTPROD tiles
  mmap-friendly layer ordering
  fused gate/up and q/gate physical layouts
  autotuned tile metadata
          |
          v
native CPU graph executor
  prompt prefill path
  single-token recurrent decode path
  bounded KV and DeltaNet state
  tokenizer, sampler, and streaming API
```

### Implementation boundary

Use a pinned `llama.cpp`/GGML revision as the graph, tokenizer, sampling,
thread-pool, cache, and serving substrate. Current upstream already contains a
Qwen3.5-family graph and a CPU `GATED_DELTA_NET` operator. Clark should add:

- a small standalone ternary microkernel library with a C ABI;
- a new GGML weight type and `.cq3`/GGUF conversion bridge;
- ISA dispatch and autotuning for Clark layouts;
- kernel-level and graph-level parity tests;
- QAT export and numerical-contract tooling.

The standalone library owns the scalar oracle and can be benchmarked without
llama.cpp. The integration layer stays thin enough to rebase onto upstream.
This avoids rebuilding tokenization, sampling, HTTP serving, graph scheduling,
and generic attention machinery, while preserving control of the novel weight
format and hot CPU paths.

For 27B weights, the ideal trit payload is about 5.35 GB. A 2-bit compute image
is about 6.75 GB before scales. FP16 group-128 scales add roughly 0.42 GB if all
27B parameters use that scheme. A practical text runtime target is therefore
7-10 GB of model files and less than 16 GB resident memory at short context,
provided the lexical tensors and continuous islands are also compressed.

The canonical and compute images need not both be resident. The runtime mmaps
the compute image; the canonical artifact can be deleted after a verified
local transcode or retained as the portable source artifact.

## Research lanes

### Lane A: numerical contract and artifact

- Specify byte order, padding, group boundaries, FP16 scale rounding, checksums,
  tensor alignment, and optional sidecar discovery.
- Preserve exact trit and scale identity from QAT export through canonical pack,
  transcode, and ISA layout.
- Build a scalar C++ oracle that is slow but dependency-free.
- Compare radix-5 LUT, aligned 2-bit, and positive/negative bitplane layouts.

Exit gate: every format reconstructs identical codes and scales, and the scalar
graph matches the PyTorch packed-forward oracle within registered tolerances.

### Lane B: decode kernels first

- Optimize group-128 ternary GEMV with A16, then A8 activations.
- Fuse RMSNorm, activation quantization, residual handling, and scale emission
  where it removes material memory traffic.
- Implement the width-4 causal convolution update and FP32 recurrent Gated
  DeltaNet update without allocating per-token temporaries.
- Implement GQA decode, partial multimodal RoPE, q/k norm, the query output
  gate, and int8 KV-cache updates.
- Fuse the two MLP input projections with SwiGLU and optimize the down projection.

Exit gate: the 4B model generates the same greedy token stream as the packed
reference for a frozen prompt set, then reaches at least 70% of a measured
memory-bandwidth roofline in the dominant GEMV microbenchmarks.

### Lane C: prefill kernels

- Add ternary GEMM tiles that amortize unpacking across multiple activations.
- Add chunked Gated DeltaNet with FP32 recurrent accumulation.
- Add tiled causal GQA attention and bounded scratch allocation.
- Autotune prompt-length regimes rather than reusing decode tiles.

Exit gate: prefill is faster than the int4 control at one registered prompt
length or the ternary runtime provides a measured memory/energy advantage large
enough to justify lower prefill speed.

### Lane D: kernel-aware QAT

- Emulate exact stored FP16 scales, group padding, A8 rounding/clipping, and
  accumulation order during training probes.
- Train continuous norms as already supported by the successful 4B recipe.
- Distill against the exact BF16 parent with temporary hidden-state anchors,
  but evaluate with the native CPU graph.
- Quantize or retrain the untied embedding and head instead of hiding them as
  BF16 islands.
- Keep the DeltaNet recurrence FP32 initially; reduce it only as a separately
  gated experiment.

Exit gate: CPU-kernel-in-the-loop evaluation agrees with the training forward,
QAT materially beats PTQ at the same rate, and no frozen capability slice loses
more than five absolute percentage points against BF16.

### Lane E: portable systems work

- ARM64 baseline: Apple M5 Max, 64 GB, using NEON/DOTPROD and measured memory
  bandwidth on the current development machine.
- x86 baseline: AVX2 without assuming VNNI.
- x86 fast path: AVX-512 plus VNNI where available.
- NUMA-aware row/layer sharding for multi-socket servers.
- mmap loading, page-fault telemetry, thread affinity, huge-page experiments,
  and signal-safe generation state.

Exit gate: identical artifact codes and bounded numerical differences across
all three ISA lanes, with reproducible build and benchmark receipts.

## Experimental sequence

### Phase 0: freeze the contract

1. Pin a known-good llama.cpp revision and record its Qwen graph parity.
2. Add the scalar oracle and `.cq3` v1 container.
3. Extract exact 4B and 27B matrix-shape manifests.
4. Register correctness tolerances and a fixed microbenchmark corpus.
5. Measure STREAM-like bandwidth and dense int4/BF16 controls on each host.

### Phase 1: 4B decode engine

1. Implement ARM64 GEMV first on the available M5 Max.
2. Bring up one DeltaNet layer, one full-attention layer, and one MLP block.
3. Execute all text layers with scalar fallbacks for unfinished operators.
4. Remove fallbacks in profile order until the full graph is useful.

### Phase 2: 4B end-to-end proof

1. Run frozen KL/perplexity and MATH, GSM8K, MMLU, and HumanEval-style gates.
2. Compare BF16, int4, radix-5-transcoded ternary, and 2-bit-native ternary.
3. Record model bytes, peak RSS, cold/warm load, prefill, decode, and energy.
4. Lock the kernel numerics that training must emulate.

### Phase 3: CPU-aware 4B QAT

1. Re-run the successful global-QAT treatment with exact deployment numerics.
2. Ablate A16 versus A8, lexical tensor formats, and recurrent precision.
3. Promote only changes that improve capability per byte and per joule.

### Phase 4: 27B scale-up

1. Run global QAT on a sharded GPU node after the runtime contract is frozen.
2. Export directly to `.cq3` plus ISA sidecars.
3. Validate first on the M5 Max 64 GB, then x86 AVX2 and AVX-512/VNNI.
4. Publish claims only after paired CPU measurements against the exact int4
   and BF16 controls.

## Scorecard

Every promoted build records:

| Axis | Required measurements |
| --- | --- |
| Correctness | code/scale hashes, layerwise max/mean error, greedy-token parity |
| Quality | parent KL/perplexity, MATH, GSM8K, MMLU, HumanEval-style execution |
| Storage | canonical bytes, compute-image bytes, continuous-island ledger |
| Memory | peak RSS, KV bytes/token, recurrent state, scratch peak |
| Speed | cold/warm load, TTFT, prefill tok/s, decode tok/s at 1/2/4/8 threads |
| Efficiency | package energy and joules/token where hardware counters permit |
| Portability | scalar, ARM64, AVX2, AVX-512/VNNI results from the same artifact |

Initial 27B product targets are a sub-10-GB text artifact, less than 16 GB RSS
at short context, and at least 10 decode tokens/s on the M5 Max reference host;
20 tokens/s is the stretch target. These are engineering targets, not claims,
and will be revised only from measured rooflines and end-to-end profiles.

## Immediate work order

1. Write the `.cq3` v1 container and scalar C++ oracle.
2. Pin and baseline the upstream llama.cpp Qwen CPU graph.
3. Build a shape-driven ternary GEMV benchmark for the exact Qwen matrices.
4. Benchmark three compute layouts on M5 Max: radix-5 LUT, aligned 2-bit, and
   dual bitplanes.
5. Implement ARM64 A16 correctness, then A8 DOTPROD.
6. Profile and optimize the upstream single-token DeltaNet state update.
7. Load the 4B packed checkpoint through the GGML graph.

No 27B training spend is justified before items 1-7 produce a reproducible
full-graph receipt.

## External technical references

- [Qwen3.8-27B model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
  is authoritative for the dimensions and layer pattern in this document.
- [Hugging Face Qwen3.5 architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md)
  documents the 3:1 Gated DeltaNet/full-attention stack and its distinct
  recurrent decode and chunked-prefill paths.
- [Microsoft bitnet.cpp](https://github.com/microsoft/BitNet) is the primary
  external control for CPU ternary GEMV/GEMM, LUT layouts, and ISA-specific
  performance. Clark still requires numerical and end-to-end validation on the
  Qwen hybrid graph rather than assuming BitNet kernel results transfer.
- [llama.cpp Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
  is the intended graph/runtime substrate. It is pinned and validated rather
  than followed at a floating branch tip.
