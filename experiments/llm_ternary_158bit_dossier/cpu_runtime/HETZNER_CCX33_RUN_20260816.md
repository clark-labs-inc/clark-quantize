# Hetzner CCX33 CPU Bring-up and Controls

Status: full 27B arithmetic pack complete; first AVX2 K2 microkernel promoted
Date: 2026-08-16
Source branch: `codex/qwen38-base3-research`
Launch commit: `dc61ec9`

## Scope and evidence boundary

This record covers node provisioning, CPU environment validation, exact parent
staging, a 4B stock llama.cpp control, the completed resumable 27B base-3
exporter, and the first native AVX2 ternary GEMV gate. The one-prompt control is
not a capability evaluation. Kernel promotion is not graph-level quality or an
end-to-end model claim.

Bulky weights, GGUF files, packed chunks, and raw process output remain on the
authorized `quant` node under `/workspace`. Git retains only source, protocols,
and this compact report.

## Node and environment

- Ubuntu 26.04 LTS, fully upgraded on 2026-08-16; no reboot was required.
- AMD EPYC Milan virtual CPU with four physical cores and eight SMT threads.
- 30 GiB usable RAM, 225 GiB root filesystem, AVX2/FMA, one NUMA node.
- SSH effective policy: public-key authentication enabled, password and
  keyboard-interactive authentication disabled, root keys permitted.
- Python 3.13.15 in `/workspace/venvs/quant-cpu`.
- CPU-only PyTorch 2.13.0, Transformers 5.15.0, Accelerate 1.14.0,
  Hugging Face Hub 1.27.0, safetensors 0.8.0, NumPy 2.5.2.
- llama.cpp commit `4df29be4f4c3673f428170fda944a5b19f743bb8`, built native with
  GCC 15.2.0 and OpenBLAS.
- Environment receipt: `/workspace/quant-state/cpu_environment.json`.
- All 84 `clark_quantize/test_*.py` tests passed in 19.68 seconds. The Linux
  native arithmetic decoder was built and loaded.

## Pinned parents

| Parent | Revision | Shards | Safetensors bytes |
| --- | --- | ---: | ---: |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 2 | 9,319,828,096 |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 18 | 55,563,006,776 |

Every index-referenced shard exists. Config and index hashes, revisions, counts,
and bytes are recorded in `/workspace/quant-state/qwen_parent_models.json`.

## Qwen3.5-4B CPU controls

The Transformers BF16 parent loaded with 8,860,648 KiB peak RSS and greedily
continued `The exact value of 2 + 2 is` with `4. This` over four generated
tokens. Generation took 7.28 seconds. This is a stack-correctness canary only.

The same parent was converted through the pinned llama.cpp Qwen3.5 hybrid graph
and then quantized with stock `TQ1_0`:

| Control | File bytes | Peak RSS KiB | Prompt-32 tok/s | Decode-16 tok/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 8,665,620,192 | 8,607,440 | 16.955 | 4.022 |
| stock `TQ1_0` | 1,314,631,392 | 1,501,572 | 17.317 | 17.843 |

llama.cpp labels the matrix format `TQ1_0 - 1.69 bpw ternary`, but reported
whole-model storage is 2.41 bpw because lexical and other protected tensors are
not all stored at that rate. On the deterministic eight-token canary, BF16
produced a coherent reasoning prefix while the stock ternary control produced
multilingual gibberish. That negative result validates the native graph and
speed control but rejects raw TQ1_0 PTQ as a quality solution.

The compact control receipt is
`/workspace/quant-state/qwen35_llama_cpu_control.json`. A failed CLI attempt
without single-turn mode entered an empty interactive prompt loop; its exact
process chain was stopped and only the accidental 77 MB prompt log was removed.

## Qwen3.8-27B CPU packing gates

The CPU-enabled streamed preflight completed in 1.65 seconds at 382,540 KiB
peak RSS. Its current mapped-weight ledger is:

| Ledger item | Value |
| --- | ---: |
| Total parameters | 27,781,427,952 |
| Ternary payload parameters | 26,869,760,000 |
| Retained high-precision parameters | 911,667,952 |
| FP16 scale count | 209,920,000 |
| Complete ideal-base-3 payload bytes | 7,566,621,155 |
| Effective bits per total parameter | 2.178900571 |

This mapping is therefore not a complete 1.58-bit model. The embedding and head
are included in the ternary payload, while excluded vision, MTP, convolution,
and other tensors keep the complete rate above two bits per original parameter.

The one-million-real-trit arithmetic reference promoted:

- exact encode/decode equality and CRC-protected frame;
- 1.584964 payload bits/trit, only 0.000001499 above `log2(3)`;
- 1.119 million encoded trits/second and 63.831 million decoded trits/second;
- reference serial full-checkpoint projection: 6.669 hours.

The full exporter completed under transient service
`quant-qwen38-base3-full.service` with four workers, four maximum in-flight
chunks, row batch 128, a 28 GiB memory cap, and no restart. It wrote and
independently decoded all 34,344 batches and 26,869,760,000 mapped trits. The
complete run took 7,661.90 seconds wall time at an effective 3.507 Mtrit/s; its
parallel independent decode took 118.18 seconds. The exact arithmetic payload
is 1.584963804 bits/trit. Once 419,840,000 bytes of FP16 scales and
1,823,335,904 referenced high-precision bytes are counted, the honest payload
is 7,567,508,848 bytes or 2.179156193 bits per original model parameter.

Runtime paths:

- progress: `/workspace/quant-state/qwen38-base3-full-progress.json`;
- final receipt: `/workspace/quant-state/qwen38-base3-full.json`;
- artifact: `/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q`.
- final receipt SHA-256:
  `ef6b8608411ba48abc85b0da390e3e8dec2b8cfad8290739c0c17557bf14cef5`.

## Radix-5 K0 control

After repairing the CPU sampler's missing explicit device argument, the sealed
10,000,000-real-trit gate passed exactly. Five-trits-per-byte encoding measured
178.670 Mtrit/s and decoding measured 472.594 Mtrit/s. The 1.6-bit code stream
costs 50,506,750 projected bytes over ideal arithmetic coding, moving the honest
whole-model projection from 2.179156 to 2.193445 bits per parameter. K0 is fast
enough that hot GEMV, not artifact transcode, is the next bottleneck.

## First K2 AVX2 kernel

The native C11 library defines group-128 activation quantization, scalar and
AVX2 A8-by-ternary GEMV, exact canonical-to-compute-image transcode, and a
checksumed STREAM-like read control. The measured matrix is the real
`model.language_model.layers.0.mlp.gate_proj.weight` tensor with shape
`17408 x 5120`; its codes and FP16 scales are authenticated directly from the
completed artifact. The group-transposed 2-bit compute image stores four
logical positions per byte and costs exactly 2.125 bits/weight including one
FP16 scale per 128 weights.

The preregistered iterations retained failures instead of moving the target:

| Iteration | Main change | Four-core GB/s | Read roofline | Scalar speedup | Promoted |
| --- | --- | ---: | ---: | ---: | --- |
| v0 | scalar table expansion | 5.976 | 17.20% | 5.97x | no |
| v1 | group-transposed vector unpack | 12.055 | 34.31% | 23.74x | no |
| v2 | persistent, physical-core workers | 21.607 | 60.55% | 22.15x | no |
| v3 | F16C and lane-wise FP32 accumulation | 25.378 | 68.35% | 32.19x | no |
| v4 | vectorized FP16 scale conversion | 25.595 | 67.05% | 31.25x | no |
| v5 | FMA, Milan tuning, affinity set once | 28.061 | 79.44% | 33.25x | yes |

The v5 output is exact across one, two, and four physical cores. Its maximum
absolute difference from the sequential scalar accumulation was
`7.450580596923828e-7`, below the sealed `5e-5` limit. The v5 receipt SHA-256 is
`985400fd4c066af40d436f695e3f2f248bb0673b24b6bfee97c02bbf85c97f37`.
This promotes the K2 microkernel only. It does not establish packed graph
parity, capability retention, end-to-end token speed, energy advantage, or a
scientific breakthrough.

The final local and Linux suites each pass 94 tests; the non-AVX2 local host
skips two ISA-specific assertions. The full Linux run also exposed and fixed a
spot-harness descriptor race: heartbeat helpers now close inherited stdio, and
the atomic status writer fsyncs its file, history, and parent directory instead
of relying on an unbounded machine-wide `sync`.

## Next gate

Integrate the promoted K2 numeric and layout contract as a narrow GGML weight
type, run full-layer parity, then load the 4B Qwen graph and compare paired
BF16, Q4, TQ1_0, and Clark ternary execution. Behavioral 27B BF16-versus-packed
evaluation still does not fit this 32 GB node and requires a larger-memory CPU
host or the high-memory GPU lane.
