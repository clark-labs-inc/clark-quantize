# Qwen3.8-27B storage layout v1

Status: **the frozen byte-accounting gate passed**. The authoritative projection
is not an exported model and has no quality or runtime promotion.

Update: the prerequisite 402-tensor 4-bit scale-codebook fit subsequently
**failed all three frozen reconstruction thresholds**. Global normalized RMSE
was 4.2785% (2% maximum), worst-tensor normalized RMSE was 10.9136% (5%
maximum), and worst per-tensor p99 relative error was 16.6767% (10% maximum).
Therefore the projected 1.624562-bpw layout is a valid storage calculation but
is not an export candidate. The remaining 264-tensor fit and complete-container
export are not promoted. See
`receipts/qwen38-27b-scale-codebook4-v1-summary.json`.

Receipt: `receipts/qwen38-27b-storage-budget-v1.json`, SHA-256
`1e37e00ff6736e6614d77f7dd1a1f69aecae96503e1659b0d027df9a37caec92`.

## Why the previous 1.585-bit headline was incomplete

The completed arithmetic-coded Qwen3.8-27B artifact stores ternary codes close
to `log2(3)`, but its complete ledger is **2.179156 bpw**. It retains
911,667,952 parameters in BF16 and stores 209,920,000 FP16 group scales. Code
entropy alone is therefore not the deployable model size.

The exact checkpoint contains 27,781,427,952 parameters in 1,199 tensors. A
whole-checkpoint ternary representation must cover the language model, visual
tower, MTP module, embeddings, head, and convolutional weights. Flattening every
rank-2-or-higher tensor by its first dimension covers 27,780,313,088 parameters
in 666 tensors. Only 1,114,864 rank-1 parameters remain exact.

## Candidate byte contract

`base3_group128_codebook4` keeps group-128 ternary granularity:

- a full group has 128 trits and is represented by a canonical integer in
  `[0, 3^128)`, requiring exactly 203 bits;
- each group has a 4-bit index into a 16-entry FP16 scale codebook belonging to
  that tensor;
- every tensor code section, scale section, and retained tensor is aligned to
  64 bytes;
- rank-1 tensors retain their authenticated source dtype;
- the tensor table counts a 64-byte fixed record, UTF-8 name, uint64 dimensions,
  global header, footer, and alignment; and
- every top-level non-shard model file is included byte-for-byte in the complete
  bundle projection.

The frozen accounting gate was at most **1.63 complete-bundle bpw**. The exact
result, produced from safetensors headers without reading weight payloads, is
**5,641,581,820 bytes / 1.6245620865 bpw**. It counts 5,507,270,842 code bytes,
108,782,072 scale-index/codebook bytes, exact rank-1 payloads, all alignment,
156,352 bytes of tensor/layout metadata, and 23,108,348 auxiliary-file bytes.

| Layout | Complete bundle bpw | Interpretation |
|---|---:|---|
| base3 group-128 + FP16 scales | 1.718507 | same group granularity, full scales |
| base3 group-128 + 6-bit scale codebooks | 1.640235 | misses the 1.63 target |
| base3 group-128 + 4-bit scale codebooks | **1.624562** | selected projected candidate |
| base3 group-512 + FP16 scales | 1.624876 | changes quantization granularity |
| base3 per-row + FP16 scale | 1.595435 | near-limit but high quality risk |
| radix-5 group-128 + FP16 scales | 1.757578 | directly byte-decodable control |
| aligned-2 group-128 + FP16 scales | 2.132538 | CPU compute-image control |

## Hard lower bound

`log2(3) = 1.5849625` bpw applies only to ternary symbols. Any deployable model
also needs scales, retained non-matrix parameters, layout metadata, and runtime
assets. Consequently a complete representation cannot truthfully equal or beat
1.5849625 bpw unless some of those costs are omitted from the denominator or
ledger. The receipt reports the no-scale entropy value only as an explicitly
unattainable lower bound.

## Promotion sequence

1. implement exact 203-bit group pack/decode and per-tensor scale-codebook fit;
2. quantify scale-codebook reconstruction error on every Qwen3.8-27B tensor;
3. export an authenticated artifact and compare measured bytes to the projection;
4. run quick CUTE and MMLU-type quality gates; and
5. benchmark direct decode versus a transient aligned-2 compute image before any
   end-to-end CPU throughput claim.

A budget pass is engineering feasibility only. It is not model-quality or
breakthrough evidence.
