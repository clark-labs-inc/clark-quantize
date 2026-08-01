# Glyph-safe promotion policy v5 amendment

Research dossier: `experiments/swin2sr_super_resolution_1_58bit/`

Policy v5 supersedes v4 only for the identity of the browser runtime harness.
The current `full_onnx_smoke_browser.html` accepts explicit fixture, model,
ONNX Runtime module, and WASM-root query parameters. This allows the identical
source to run from an isolated local HTTP root without embedding one
workstation's paths.

The source change does not alter model execution, numerical tolerances,
timing collection, browser-family detection, backend selection, or receipt
fields. Its SHA-256 changed from
`299bdb8e9b277a395c1416dc1ebe9984a9455a375b78cdf0e83c8b253e762e4e`
to
`d2939e8ae632701bcfbaa47e1cc3608840886e5d40cdcdac0b47d5f951947a5f`.

All selection and acceptance conditions remain byte-for-byte equivalent after
removing the v5 `amendment` object and substituting the old harness hash:

- the exact q4 model, revision, artifact bytes, and state hash;
- the locked 3,040-sample glyph-safe bundle;
- paired mean, p10, source, script, and language quality gates;
- the strict smaller-artifact gate;
- Chromium WebGPU plus Chromium, Firefox, Safari, iOS Safari, and Android
  Chrome WASM surfaces;
- the `[1, 3, 128, 128]` runtime input;
- every numerical and latency limit.

The q4 metrics do not require recomputation because neither the reference
artifact nor the quality/evaluation contract changed. Policy v4 is retained as
the immutable superseded record.
