# Hetzner CCX33 CPU Bring-up and Controls

Status: full 27B arithmetic pack running
Date: 2026-08-16
Source branch: `codex/qwen38-base3-research`
Launch commit: `dc61ec9`

## Scope and evidence boundary

This record covers node provisioning, CPU environment validation, exact parent
staging, a 4B stock llama.cpp control, and launch of the existing resumable 27B
base-3 exporter. The one-prompt control is not a capability evaluation. The
running full export is not reported as complete until its final receipt passes.

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

The full exporter was launched as transient service
`quant-qwen38-base3-full.service` with four workers, four maximum in-flight
chunks, row batch 128, a 28 GiB memory cap, and at most three restart attempts.
At the recorded launch snapshot it had committed 60 of 34,344 batches,
regenerated 41,287,680 trits in 10.43 seconds, used about 0.90 GB peak service
memory, and had not restarted. That early rate suggests roughly 1.8 hours, but
the final receipt is authoritative.

Runtime paths:

- progress: `/workspace/quant-state/qwen38-base3-full-progress.json`;
- final receipt: `/workspace/quant-state/qwen38-base3-full.json`;
- artifact: `/workspace/quant-runs/qwen38-base3-full/artifacts/qwen38-optimal.cq3q`.

## Next gate

Wait for the final arithmetic-pack receipt and independent full decode. If it
promotes, benchmark the 1.6-bit radix-5 compute/distribution control and begin
the 4B native Clark weight-type integration. Behavioral 27B BF16-versus-packed
evaluation does not fit this 32 GB node and must use a larger-memory CPU host or
the existing high-memory GPU lane.
