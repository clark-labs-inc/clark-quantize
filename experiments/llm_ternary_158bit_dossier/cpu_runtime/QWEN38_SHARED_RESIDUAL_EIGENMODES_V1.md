# Shared Residual Eigenmodes for Qwen3.8-27B

Status: subordinate target-compression candidate; frozen pilot required before fit

## Purpose

The target is not minimum bits. The target is a nearly lossless Qwen3.8-27B
model that materially improves AMD CPU decode throughput over Q4_K_M. Full
TQ2_0 shows that the current runtime can reach 3.73 tokens/s on four physical
cores, but raw ternarization destroys capability. The approximation therefore
spends a small, CPU-friendly residual branch only on activation modes that the
two-bit matrix cannot represent.

This is not the primary route to the current 100 accepted tok/s objective.
Even a successful pilot can only qualify a smaller target model for
multi-token verification. It cannot promote without exact speculative
semantics and the end-to-end contract in `qwen38_27b_amd_100tps_goal_v1.json`.

## Projection-bundle approximation

Several Qwen projections consume the same activation vector. For a bundle of
such matrices, write

```text
W_i = Q_i + E_i
W_i x ~= Q_i x + A_i (B x),  i = 1,...,k
```

`Q_i` is the exact deployed aligned-two-bit/TQ2 matrix. `B` is one shared
right factor for the bundle and `A_i` is an output-specific left factor. The
expensive input projection `z = B x` is computed once. In the MLP, gate and up
share `z`; their corrected outputs enter the ordinary SwiGLU path. Down has a
separate factor because its input space differs.

For calibration activations `X` and vertically stacked residual
`E = [E_1; ...; E_k]`, the continuous rank-`r` initialization solves

```text
min rank(R) <= r  ||(E - R) X||_F^2.
```

Let `Sigma = XX^T / n + lambda I` and let

```text
E Sigma^(1/2) = U S V^T.
```

The weighted Eckart-Young solution is

```text
R_r = U_r S_r V_r^T Sigma^(-1/2),
A = U_r S_r,
B = V_r^T Sigma^(-1/2).
```

This is the best continuous shared rank-`r` correction under the registered
activation metric. It is not itself a novelty claim: activation-aware SVD,
low-rank-plus-quantized decomposition, joint weight-activation subspaces, and
fused low-rank residual branches all have prior art. The research question is
whether the same-input bundle, exact TQ2 forward, hard INT8 residual state, and
AMD fused execution form a useful new Pareto point for this hybrid Qwen graph.

## Exact deployment algebra

The pilot exports only:

- upstream-compatible TQ2 base blocks and their scales;
- group-128 symmetric INT8 `B` and `A_i` factors with stored FP16 scales;
- one bundle descriptor connecting a shared `B` to its consumers;
- no BF16 residual, dense correction, optimizer state, or reconstruction
  cache.

At runtime, the existing A8 activation is reused for `Q_i x` and `B x`. The
small latent vector is requantized once, then each `A_i z` is accumulated into
the corresponding projection output. A fused gate/up kernel may share input
loads, `B x`, latent quantization, and the output epilogue. FP32 is retained for
the DeltaNet recurrent state, norms, and registered numerically sensitive
nonlinearities.

With rank 64, the layer-0 gate/up residual has about 2.56 MB of INT8 factor
payload before scales, versus about 178 million base weights. Its extra dot
products are about 2.56 million per token, roughly 1.4% of the two base
projections' logical weights. These estimates are admission calculations, not
measured speed.

## Why the earlier rank-40 result does not answer this question

The rejected rank-40 binary-factor experiment optimized weight reconstruction
under a 1.55-bit ceiling. It reduced squared weight error by only 1.25% versus
direct ternary. This candidate changes all three causal assumptions:

1. the objective is complete-block output under real activations;
2. factors are INT8 rather than binary/INT4 and the bit ceiling is removed;
3. the base is the fast TQ2 compute path and residual compute is shared by
   same-input projections.

The old negative result remains valid for its exact binary rank-40 family.

## Stop condition

The fixed three-block pilot must materially reduce held-out complete-block
error on every selected depth and demonstrate a credible same-shape AMD kernel
speed advantage over Q4. Failure closes this exact rank-64 INT8 shared-mode
family. Success authorizes a restartable whole-model materialization followed
immediately by free-generation and multi-token target-verification gates. A
block-error or single-token kernel pass is not evidence for 100 tok/s.
