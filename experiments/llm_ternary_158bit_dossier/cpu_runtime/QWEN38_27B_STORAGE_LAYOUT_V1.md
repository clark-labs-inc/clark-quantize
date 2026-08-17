# Qwen3.8-27B storage layout v1

Status: projected layout awaiting the frozen authoritative inventory run. It is
not an exported model and has no quality or runtime promotion.

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

The frozen accounting gate is at most **1.63 complete-bundle bpw**. The exact
result is produced from safetensors headers without reading weight payloads.

## Hard lower bound

`log2(3) = 1.5849625` bpw applies only to ternary symbols. Any deployable model
also needs scales, retained non-matrix parameters, layout metadata, and runtime
assets. Consequently a complete representation cannot truthfully equal or beat
1.5849625 bpw unless some of those costs are omitted from the denominator or
ledger. The receipt reports the no-scale entropy value only as an explicitly
unattainable lower bound.

## Promotion sequence

1. reproduce the frozen inventory and byte ledger;
2. implement exact 203-bit group pack/decode and per-tensor scale-codebook fit;
3. quantify scale-codebook reconstruction error on every Qwen3.8-27B tensor;
4. export an authenticated artifact and compare measured bytes to the projection;
5. run quick CUTE and MMLU-type quality gates; and
6. benchmark direct decode versus a transient aligned-2 compute image before any
   end-to-end CPU throughput claim.

A budget pass is engineering feasibility only. It is not model-quality or
breakthrough evidence.
