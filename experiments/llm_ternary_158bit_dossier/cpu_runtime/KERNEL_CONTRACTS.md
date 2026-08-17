# Qwen CPU Kernel Contracts

Status: implementation contract v1
Date: 2026-08-16

These are operator contracts, not implementation promises. Each optimized
kernel must retain a scalar reference and must be testable independently of the
full model.

The optimized primitives live in a small C-ABI library and are integrated as a
GGML weight type and operator path in a pinned llama.cpp revision. Upstream
Qwen3.5 CPU operators are correctness and performance controls; they are
replaced only where profiling shows a material end-to-end gain.

## Common numeric contract

- Weight codes are exactly `{-1, 0, +1}`.
- Matmul weights use groups of 128 consecutive input features.
- Each group has one stored little-endian FP16 scale.
- The A16 reference accumulates group partials in FP32.
- The A8 path uses group-128 max-absolute scales, `lrintf` under the default
  round-to-nearest mode, clipping to `[-127, 127]`, and a unit scale for an
  all-zero group. Integer dots are exact. The promoted AVX2 path accumulates
  eight lanes in FP32 with FMA across groups and performs one fixed pairwise
  reduction per output row; deployment-aware QAT must emulate this order.
- Tail groups are zero-padded and cannot change the logical tensor shape.
- Optimized kernels must consume the same codes and scales as the scalar oracle.

## Kernel inventory and order

| ID | Kernel | Decode | Prefill | First optimized ISA | Priority |
| --- | --- | --- | --- | --- | ---: |
| K0 | canonical pack/transcode/verify | load | load | portable scalar | 0 |
| K1 | RMSNorm + A8 quantize + residual staging | every block | every block | x86 AVX2 | 0 |
| K2 | group-128 ternary GEMV | dominant | fallback | x86 AVX2/F16C/FMA | 0 |
| K3 | group-128 ternary GEMM | small batches | dominant | x86 AVX2 | 1 |
| K4 | fused gate/up projection + SwiGLU | every block | every block | ARM64 | 1 |
| K5 | depthwise causal-conv4 update | 48 layers | 48 layers | ARM64 | 0 |
| K6 | recurrent Gated DeltaNet update | 48 layers | tail | ARM64 | 0 |
| K7 | chunked Gated DeltaNet | no | 48 layers | portable tiled | 1 |
| K8 | GQA attention + partial RoPE + q/k norm + q gate | 16 layers | 16 layers | ARM64 | 1 |
| K9 | KV quantize/dequantize and paged storage | 16 layers | 16 layers | ARM64 | 1 |
| K10 | embedding lookup and LM-head logits/top-k | once/token | once/token | ARM64 | 1 |
| K11 | GGML integration, mmap, thread pool, NUMA scheduling | whole graph | whole graph | portable | 0 |

## K0: canonical pack and compute-image transcode

Input is canonical radix-5 codes, FP16 scales, tensor metadata, and checksums.
Output is an aligned, mmap-safe compute image containing the same logical codes
and scales. Candidate physical layouts are:

1. five-trit bytes plus register LUTs;
2. group-128-transposed 2-bit codes, where byte `j` stores logical positions
   `j`, `j+32`, `j+64`, and `j+96`;
3. separate positive and negative bitplanes.

The transcode receipt includes source and output hashes, selected ISA, tile
shape, padding count, and an exact decoded-code comparison. Runtime selection
is by measured matrix families, not one global winner assumed in advance.

## K1-K3: normalization and ternary linear algebra

K1 emits both the normalized floating activation and, where enabled, an int8
activation plus its scale. This prevents every projection in a block from
re-reading and requantizing the same input.

K2 specializes single-token matrix-vector work. It should fuse group-scale
application into FP32 accumulation and avoid materializing unpacked weights.
The first promoted AVX2 implementation uses A8 activations, an authenticated
2.125-bit/weight compute image including FP16 scales, and reaches 79.44% of the
measured four-core memory-read roofline on the CCX33 Milan control. This is a
microkernel result. The same code/scale contract is now integrated as GGML
`CQ3_0`/`CQ8_0`, including `MUL_MAT` and embedding `GET_ROWS`, and executes the
full Qwen3.5-4B graph. That runtime milestone does not promote the PTQ model:
the frozen perplexity, MATH-500, code, and CUTE tripwire rejected its quality.
QAT-to-runtime numerical agreement remains a separate gate.
The first ARM64 implementation uses the identical aligned-2 compute image and
NEON signed DOTPROD. On the real Qwen3.8-27B `17408 x 5120` gate projection it
is 28.38x faster than the scalar oracle, sustains 60.39 GB/s with four M5 Max
workers, and reaches 67.20% of the paired NEON read roofline with maximum
absolute error `7.15e-7`. The preregistered ARM v1 milestone passed, but the
70% K2 exit gate and full-graph gate remain open. Eight workers add no
throughput, motivating native fused gate/up dispatch rather than more Python
worker fan-out.
The benchmark matrix families are the exact Qwen shapes:

- `5120 -> 17408` and `17408 -> 5120` for the MLP;
- `5120 -> 10240`, `5120 -> 6144`, and `6144 -> 5120` for DeltaNet;
- `5120 -> 12288`, `5120 -> 1024`, `6144 -> 5120` for full attention;
- `5120 -> 248320` for the LM head.

K3 specializes prompt batches and amortizes decode/unpack work across activation
rows. Its tile search is separate from K2. At minimum test prompt blocks of 8,
32, 128, and 512 tokens.

## K4: fused SwiGLU

The gate and up weights are physically adjacent or interleaved so one scheduled
operation produces both projections. Apply SiLU to the gate, multiply by the up
activation, and write only the final 17,408-wide intermediate. Do not duplicate
activation quantization or store two avoidable full-width temporary buffers.
The first ARM64 native operator now executes both real Qwen3.8-27B layer-0
projections and SwiGLU with bit-exact agreement against two separate DOTPROD
projections plus the same native SwiGLU control. It improves median latency by
8.1% with four M5 Max workers and 27.6% with eight, while the one-worker result
is 1.4% slower. This passes the frozen K4 operator gate; graph dispatch, token
parity, capability, and end-to-end performance remain open.

## K5-K7: Gated DeltaNet

K5 maintains a three-sample history for the width-4 depthwise convolution over
10,240 channels, applies the trained convolution weights, and applies SiLU.
The single-token path updates state in place.

K6 consumes normalized q/k, value, sigmoid beta, and FP32 decay `g`. Per value
head it updates an FP32 `128 x 128` recurrent matrix and emits one 128-wide
value. The implementation must not allocate a new 3 MiB state per layer per
token. Q and K heads are logically repeated from 16 to 48 heads, but the kernel
should index or broadcast them rather than materializing the repeats.

K7 is a distinct chunked-prefill algorithm. It uses bounded tiles and returns
the same final recurrent state as the scalar token-by-token oracle. Numerical
comparison includes both every output token and the final FP32 state.

The post-recurrence fused operation is FP32 RMS normalization over each
128-wide value head, multiplication by learned norm weights, SiLU gating from
`z`, conversion to the activation dtype, and the `6144 -> 5120` output
projection.

## K8-K9: full attention

The query projection emits 24 query heads and 24 output-gate heads. K8 performs
head-dimension RMSNorm for Q and K, applies partial multimodal RoPE, executes
causal GQA with 24 query heads and 4 KV heads, applies the sigmoid output gate,
and feeds the output projection.

K9 stores KV per full-attention layer. The first correctness format is BF16;
the product format target is int8 with explicit per-head or per-block scales.
At batch one, total BF16 KV growth is 64 KiB/token and int8 target growth is
about 32 KiB/token before scale metadata. Cache quantization is evaluated for
quality separately from ternary weight error.

## K10: lexical tensors

Embedding and LM head are untied and each contains about 1.27B parameters.
The first graph may use Q6/Q8 embedding rows and an int4/Q6 head. A ternary head
is promoted only if QAT recovers the lexical quality loss. The LM-head kernel
must fuse or stream top-k selection so it does not materialize avoidable copies
of the 248,320-way logit vector.

## K11: execution system

- reuse pinned llama.cpp graph, tokenizer, sampling, cache, and server surfaces;
- keep the Clark GGML patch narrow enough to rebase and test independently;
- mmap the compute image and report cold versus warm page faults;
- pin worker threads and separate physical-core from SMT results;
- shard weights according to NUMA ownership on multi-socket systems;
- use bounded reusable arenas for activations and attention scratch;
- expose prompt ingestion, token streaming, cancellation, and signal-safe state;
- emit per-operator time, bytes read, and selected kernel implementation.

## Required tests for every optimized kernel

1. scalar-versus-optimized randomized property tests;
2. real tensor-shape tests with tail groups and adversarial scale values;
3. exact code/scale identity after transcode;
4. deterministic output under repeated runs and thread counts;
5. finite-value and NaN propagation policy tests;
6. microbenchmarks with compiler, ISA, clocks, threads, and bandwidth receipt;
7. full-layer comparison before full-graph promotion.

Speed without graph-level quality parity is a failed kernel. Correctness without
measured end-to-end impact remains a reference implementation, not a promoted
optimization.
