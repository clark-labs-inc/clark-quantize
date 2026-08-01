# Promotion policy v9 amendment

Date: 2026-07-31
Supersedes: `promotion_policy_glyphsafe_v8.json`
Superseded SHA-256:
`1f56a1a34de8e235de40392d85e1331b3a9f6cb45fa42b3565625293b3d43c96`

## Reason

V8 pinned a fixture-aware expanded-ONNX browser harness. V9 switched the
runtime gate to the preregistered packed browser-v3 artifact: WebGPU uses the
external-data graph, while the independent packed native/SIMD-to-scalar WASM
graph covers fallback execution.

## Unchanged contract

- q4 reference identity and 3,040-sample evaluation bundle;
- paired, p10, source, script, and language quality gates;
- bootstrap seed and sample count;
- artifact-size threshold and gzip transport;
- required browser surfaces, input shape, numerical limits, and latency limits.

V9 is immutable historical evidence. V10 separately pins the later
signed-int7 v4 WebAssembly payloads used by future candidates.
