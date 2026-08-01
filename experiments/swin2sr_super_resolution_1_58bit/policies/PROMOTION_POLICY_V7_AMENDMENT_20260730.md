# Swin2SR promotion policy v7 amendment

Research dossier: `experiments/swin2sr_super_resolution_1_58bit/`

Date: 2026-07-30

`promotion_policy_glyphsafe_v7.json` supersedes v6 for the identity and
verification of the deployable model package. The full portable ONNX graph
stores every selected weight as base-3 codes plus group scales, but protobuf
graph metadata made the raw file 4.81 MB. Deterministic gzip reduces the same
graph to approximately 650 KB without changing its expanded bytes or numerical
execution.

The browser harness now:

1. downloads the `.onnx.gz` package;
2. verifies the package SHA-256 and byte count;
3. expands it locally with the browser `DecompressionStream`;
4. verifies the expanded ONNX SHA-256;
5. gives those exact expanded bytes to ONNX Runtime WebGPU or WASM;
6. records both identities in every runtime receipt.

Promotion requires `modelCompression == "gzip"` and rejects a mismatched
package, expanded graph, harness, browser, backend, input shape, numerical
result, or latency.

The v6 maximum artifact size remains unchanged at 50% of q4. The paired
quality margins, source/language/script gates, final data, q4 reference,
required runtime surfaces, numerical tolerances, and latency limits are also
unchanged.
