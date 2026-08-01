# Promotion policy v10 amendment

Date: 2026-08-01
Supersedes: `promotion_policy_glyphsafe_v9.json`
Superseded SHA-256:
`289854f2bf18097d2f5f8d0f78384c404ffc33ca4656ea70c60339403910a933`

## Reason

V9 is the immutable pre-G3 packed-runtime policy and pins its exact native and
SIMD WebAssembly bytes. The selected G3 artifact adds four signed-int7 RSTB
precision islands and therefore uses the later v4 decoder. V10 pins the exact
current scalar and SIMD WebAssembly payloads before any future candidate
evaluation instead of rewriting V9's historical hashes.

## Unchanged contract

- q4 reference model, revision, state, artifact bytes, and 3,040-sample bundle;
- all paired, p10, source, script, and language quality gates;
- 10,000 bootstrap samples and seed;
- the 50% maximum artifact-size fraction;
- gzip transport, fixture identity, and canonical-state requirements;
- all six required WebGPU/WASM browser surfaces;
- input shape, numerical tolerances, and latency limits;
- the HTML, JavaScript, and runtime-surface hashes.

This amendment updates only the native and SIMD WebAssembly payload identities.
It does not retroactively alter a V9 result or authorize q4-equivalent quality.
