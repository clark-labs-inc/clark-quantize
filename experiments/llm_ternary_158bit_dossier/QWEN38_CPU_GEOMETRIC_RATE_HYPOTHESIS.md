# Qwen3.8-27B CPU geometric-rate hypothesis

Status: preregistered research program, not a breakthrough claim.

Scope: `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` only.

## The coupled hypothesis

The deployable rate and the approximation geometry should be optimized
together.  A useful Qwen3.8-27B candidate must satisfy all three constraints:

1. Lower activation-weighted approximation error than the current symmetric
   group-128 ternary projection.
2. An honest complete-model rate near 1.58 bits per total parameter after
   scales, centers, retained tensors, metadata, alignment, and auxiliary files.
3. A CPU implementation whose transform and correction costs do not erase the
   ternary GEMV advantage.

The first frozen family combines a row-shared affine ternary manifold, an
optional block-128 randomized Hadamard transform, and source-specific lossless
coding of the resulting ternary symbols.

## Affine ternary manifold

For output row `r`, input group `g`, and position `i`, represent a weight as

```text
W[r,g,i] ~= mu[r] + alpha[r,g] q[r,g,i],  q in {-1, 0, +1}.
```

With nonnegative diagonal activation weights `h[i]`, minimize

```text
sum[r,g,i] h[i] (W[r,g,i] - mu[r] - alpha[r,g] q[r,g,i])^2.
```

For fixed codes, define per row and group

```text
H_g = sum h_i                 C_g = sum h_i W_i
A_g = sum h_i q_i^2           B_g = sum h_i q_i
D_g = sum h_i q_i W_i.
```

After analytically eliminating each group scale, the exact row center is

```text
mu = sum_g (C_g - B_g D_g / A_g)
     --------------------------------,
     sum_g (H_g - B_g^2 / A_g)
```

with the natural `A_g = 0` limits.  Then

```text
alpha_g = (D_g - mu B_g) / A_g.
```

For fixed `mu` and nonnegative `alpha`, every code is independently assigned to
the nearest of `mu-alpha`, `mu`, and `mu+alpha`.  Alternating the exact
parameter solve and exact nearest-level assignment cannot increase the frozen
objective except for bounded floating-point roundoff.  The symmetric baseline
codes are retained as one deterministic initialization, so the FP32 fit has a
no-regression starting point.

The CPU identity is

```text
y_r = sum_g alpha[r,g] dot(q[r,g,:], x[g,:]) + mu[r] sum_i x_i.
```

The existing digit-centered ternary path already forms activation sums.  The
row center therefore adds one FP16 value per output row and approximately one
row FMA, rather than a dense residual.

## Block randomized Hadamard geometry

For each 128-column group, let

```text
R = D H / sqrt(128),
```

where `D` is a deterministic Rademacher diagonal and `H` is a Walsh-Hadamard
matrix.  Since `R R^T = I`,

```text
W x = (W R) (R^T x).
```

The dense model is therefore unchanged before quantization.  Ternarization is
performed in the rotated coordinates.  At inference, the corresponding
activation transform uses only add/subtract butterflies.  The same transformed
activation can be reused by projections that share an input.  The frozen seed
regenerates `D`, so no sign payload is required.

This is a CPU-constrained variant of incoherence processing, not a claim that
randomized Hadamard quantization is new.  QuIP and QuIP# established the value
of incoherence processing for low-bit quantization, while this experiment asks
whether a block-128 transform plus an affine ternary line produces a favorable
Qwen3.8 CPU rate/quality/cost point.

## Why `log2(3)` is not the model-specific storage floor

`log2(3)` is the rate of a uniform ternary source and the source-agnostic
worst-case target used by the existing canonical packer.  It is not a lower
bound for the fixed, nonuniform codes of one trained model.

For a block containing counts `(n_-, n_0, n_+)`, the number of sequences with
that composition is

```text
M = n! / (n_-! n_0! n_+!).
```

An enumerative code stores the sequence rank in `ceil(log2 M)` bits once the
composition is known.  The type-class inequality gives

```text
log2 M <= n H(n_-/n, n_0/n, n_+/n).
```

The existing one-million-code Qwen3.8 sample has counts
`(266670, 466137, 267193)`, empirical entropy `1.5305514` bits/code, and a gap
of `0.0544111` bits/code below uniform base-3.  If the authenticated full code
set confirms a similar distribution, that gap is large enough in projection
to pay for row centers and move the complete row-normalized layout from about
1.628 to about 1.575 bits per total parameter.

That extrapolation is not evidence.  The frozen full-code analysis must decode
and hash all 34,344 frames and all 26,869,760,000 mapped codes, retain uniform
base-3 for the unmapped 264 tensors, charge a 16-byte header and 64 numerical
guard bits per chunk, byte-align every chunk, and 64-byte-align every tensor.

## Frozen decision sequence

1. Finish the current CUTE, MATH/code, MMLU, and WikiText capability transaction.
2. Run the 12-span real-weight geometric ablation over layer-0 gate, up, and
   down projections.  A pass requires at least 0.5% aggregate squared-error
   reduction and no more than 0.005 added entropy-plus-center bits/weight.
3. Authenticate the complete mapped ternary-code distribution.  A pass requires
   a conservative complete projection no larger than 1.59 bits/parameter and
   at least 0.04 effective bit/code savings after chunk headers.
4. If geometry passes, repeat it with real activation second moments and full
   layer-output error, then integrate it into bounded layer-0 QAT.
5. If the rate bound passes, implement and round-trip an actual enumerative or
   static-probability codec, followed by native load-time transcode measurements.
6. Export only a candidate that survives CUTE, MMLU-type, MATH, WikiText, and
   paired CPU prefill/decode throughput gates.

## Clark Hash as a trained residual path

Clark Hash observes that a stored vector and a floating query can be scored
asymmetrically after the same deterministic sparse signed projection.  A linear
layer has the same algebra: weight rows are the stored vectors and the current
activation is the query.  With a sparse projection `R`, the direct estimator is

```text
W x ~= (W R^T) (R x).
```

The activation sketch `R x` is formed once and shared by every output row.  The
paper's SplitMix64 bucket/sign construction is cheap and stateless, which is
attractive for CPU inference.

This direct estimator is not semi-lossless.  For `m < d`, the sketch has rank at
most `m` and cannot reconstruct an arbitrary dense row.  Johnson-Lindenstrauss
preservation applies to a fixed finite set of vectors or inner products with a
probabilistic distortion, not to exact recovery of every future activation.

The frozen hybrid under test is instead

```text
W ~= T + A R,
y  = T x + A (R x),
```

where `T` is a rate-constrained ternary residual and `A` is a small correction
matrix.  For fixed `T`, the data-free coefficient solve is

```text
A* = (W - T) R^T (R R^T + lambda I)^-1.
```

For fixed `A`, the best ternary residual with exactly `k` nonzeros per group
uses the top-`k` residual magnitudes and their mean absolute value as the group
scale.  These two updates form a monotone FP32 coordinate fit.  Export then
quantizes `A` to signed per-row int4 and `T` scales to FP16 for the first screen.

At Qwen input width 5120, one frozen candidate uses a sketch of 80 coordinates
and 60 nonzero ternary values per 128-weight group.  Its projected component
rate is about 1.566 bits/weight when counting empirical ternary entropy, int4
hash coefficients, one FP16 hash scale per row, a row-normalized 4-bit ternary
scale layout, and both row side channels.  This is only a component projection;
the first error screen still uses FP16 ternary group scales and cannot promote a
complete storage claim.

If the data-free screen passes, the next training step replaces Frobenius least
squares with calibration activations:

```text
A* = argmin_A ||(W - T) X - A (R X)||_F^2,
```

followed by a short teacher-output/KL QAT phase over the layer.  Here
"semi-lossless" is reserved for a candidate whose frozen layer outputs,
capability evals, and perplexity remain within preregistered tolerances.  It
never means byte-exact reconstruction of the original dense weights.

Clark Hash paper source: <https://github.com/clark-labs-inc/clark-hash/blob/main/arxiv_submission/main.tex>
at publication commit `de609c2c55dcbf727deea8bdf7bcc657743ce120`.

## Related-work boundary

- QuIP: <https://arxiv.org/abs/2307.13304>
- QuIP#: <https://arxiv.org/abs/2402.04396>
- TernaryLLM: <https://arxiv.org/abs/2406.07177>
- QTIP: <https://arxiv.org/abs/2406.11235>
- ParetoQ: <https://arxiv.org/abs/2502.02631>

The affine shift is related to asymmetric/learnable ternarization, the RHT is
related to incoherence processing, and source-specific entropy coding is
classical information theory.  The research contribution under test is their
joint CPU-deployable formulation and measured Pareto point on Qwen3.8-27B.
No novelty or breakthrough claim is authorized until the full frozen quality,
storage, correctness, and throughput gates pass reproducibly.
