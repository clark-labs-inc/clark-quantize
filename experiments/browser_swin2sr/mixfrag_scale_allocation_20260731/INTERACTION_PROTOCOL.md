# Combined scale-allocation interaction protocol

## Question

Do the 10,012-byte and 20,024-byte per-surface allocations selected from
isolated MixFrag-style perturbations remain better than smaller uniform
controls when every selected precision is materialized at once?

## Isolation

This is a mechanism gate, not a model-promotion evaluation. It uses the exact
64 training-only records that selected the allocation and a second deterministic
64-record training-only shard that excludes every allocation record. Development
and final-heldout access remain prohibited.

## Controls and gate

The 10,012-byte allocation is compared with uniform q2 scales. The 20,024-byte
allocation is compared with uniform q4 scales. FP16 per-group scales measure the
remaining representation floor. A mixed allocation advances to separately
sealed QAT only if it has at least 10% lower combined output fragility than its
matched uniform control on both shards, covers all 204 surfaces, reproduces its
sealed byte receipt, and fits its physical browser-aligned byte budget.

Output fragility is normalized output MSE plus 0.25 times normalized Sobel-edge
MSE against the same hard-ternary parent with uncompressed scales. The measured
combined result overrides the sum of isolated perturbations.

## Storage erratum

The original sealed preflight estimated uniform q8 scale storage at 65,228
bytes. Completed calibration produced 65,232 bytes, and the browser layout's
separately aligned center/index physical span is 65,548 bytes. This diagnostic
control is not an advancing allocation. Both advancing allocations are checked
against their actual per-surface receipts and physical aligned spans before the
study is sealed.
