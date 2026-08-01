# Swin2SR promotion policy v6 amendment

Research dossier: `experiments/swin2sr_super_resolution_1_58bit/`

Date: 2026-07-30

`promotion_policy_glyphsafe_v6.json` supersedes v5 only for deployable artifact
size. V5 required the candidate artifact to be smaller than the shipped q4
artifact but did not quantify "materially smaller." That allowed the 4.84 MB
portable ONNX compatibility graph to pass against the 5.68 MB q4 reference,
even though it was not the approximately 1.58-bit browser artifact required by
the study objective.

V6 requires the exact artifact named by every runtime receipt to be no larger
than 50% of the q4 reference:

```text
candidate runtime artifact <= 2,838,021 bytes
```

This threshold forces promotion through the compact packed graph rather than
the correctness-only portable ONNX fallback. The existing 414,092-byte compact
infrastructure smoke is below the threshold, but it is not a promotion
candidate: it still needs a selected fully hard checkpoint, complete graph
execution, and all required browser receipts.

The paired quality margins, source/language/script gates, final data,
reference model, runtime surfaces, numerical conformance, latency limits,
input shape, and runtime harness hashes are unchanged.
