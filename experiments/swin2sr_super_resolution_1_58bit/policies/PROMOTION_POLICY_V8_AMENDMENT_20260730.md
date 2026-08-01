# Promotion policy v8 amendment

Date: 2026-07-30
Supersedes: `promotion_policy_glyphsafe_v7.json`
Superseded SHA-256:
`e36cc88b3191395cd9b79ba436272f0643284703253966e13f3726ac515802a7`

## Reason

V7 binds every runtime receipt to the selected compressed package, expanded
ONNX graph, detected browser surface, backend, and runtime harness. It did not
bind the expected-output fixture to the canonical dense hard state.

V8 requires the browser harness to hash the complete fixture and record the
fixture's embedded hard-state SHA-256 on every surface. The promotion gate
rejects a receipt whose fixture is missing, changed, or derived from another
hard state.

## Unchanged contract

- q4 reference model, revision, state, artifact bytes, and final bundle;
- all paired and stratified quality gates;
- 10,000 bootstrap samples and seed;
- the 50% maximum artifact-size fraction;
- gzip package and expanded ONNX identities;
- all six required WebGPU/WASM browser surfaces;
- input shape, numerical tolerances, latency limits, and memory observations.

This is a pre-outcome provenance repair. No final candidate, final metric, or
runtime result existed when it was registered.
