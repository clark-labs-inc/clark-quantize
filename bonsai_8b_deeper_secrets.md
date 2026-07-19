# Bonsai-8B 1-Bit: Deeper Reverse-Engineering Report

## Executive Summary

This report extends the initial forensic analysis with deeper findings from:
1. Cross-model comparison across Bonsai 1.7B, 4B, and 8B (universal vs model-specific patterns)
2. Scale formula reverse-engineering (predicting Bonsai amplitude from base weights)
3. Optimizer signature analysis (singular value distributions, update orthogonality)
4. Code spatial structure (correlations, autocorrelation, FFT)
5. Training data signature (embedding clusters, vocab pruning)
6. Training step estimation (flip rate decay model)
7. The official Bonsai whitepaper (March 31, 2026)

---

## 1. Cross-Model Universality (1.7B vs 4B vs 8B)

### 1.1 Architecture

All three models share identical quantization structure:
- Same 267 vocab tokens pruned (151936 → 151669) across ALL sizes
- Same 1-bit binary g128 quantization on all 2D tensors
- Same FP16 island (1D RMSNorms only)
- Same MLX packing format

### 1.2 Universal Patterns

| Metric | 1.7B | 4B | 8B |
|---|---|---|---|
| Overall flip rate (L_mid gate_proj) | 29.68% | 28.58% | 27.46% |
| Code balance (+1%) | 50.04% | 50.09% | 49.99% |
| Scale ratio (L_mid gate_proj) | 3.00× | 2.46× | 1.97× |
| Weight delta (L_mid gate_proj) | 2.00 | 1.63 | 1.31 |

**Universal across all sizes:**
- Perfect 50/50 code balance (±0.1%)
- v_proj always has the lowest flip rate (~20-23%)
- gate_proj always has the highest flip rate (~27-30%)
- Scale ratio: up_proj > gate_proj > v_proj > o_proj > k_proj > q_proj > down_proj
- Embeddings are frozen in 1.7B and 8B (ratio=1.000, flip<0.08%)
- **Embeddings are TRAINED in 4B** (flip=6.56%, ratio=0.990)

**Scale-dependent patterns:**
- Smaller models have HIGHER flip rates (29.7% → 27.5% from 1.7B → 8B)
- Smaller models have HIGHER scale ratios (3.0× → 2.0× from 1.7B → 8B)
- Smaller models have HIGHER weight deltas (2.0 → 1.3 from 1.7B → 8B)
- This suggests smaller models need to "work harder" to compensate for binarization

### 1.3 Per-Type Flip Rate (middle layer)

| Type | 1.7B | 4B | 8B |
|---|---|---|---|
| q_proj | 28.61% | 28.85% | 26.72% |
| k_proj | 28.05% | 26.00% | 25.07% |
| **v_proj** | **22.78%** | **20.65%** | **19.96%** |
| o_proj | 25.44% | 24.40% | 23.42% |
| gate_proj | 29.68% | 28.58% | 27.46% |
| up_proj | 29.24% | 27.47% | 26.09% |
| down_proj | 27.18% | 24.47% | 23.11% |

**v_proj is universally the most stable** (lowest flip rate across all sizes). This makes sense: v_proj determines what information is read from the attention values, and its sign structure is most critical for attention semantics.

### 1.4 Per-Type Scale Ratio (middle layer)

| Type | 1.7B | 4B | 8B |
|---|---|---|---|
| q_proj | 1.73× | 1.94× | 1.55× |
| k_proj | 1.99× | 2.17× | 1.90× |
| v_proj | 2.76× | 2.59× | 2.29× |
| o_proj | 2.51× | 2.02× | 1.92× |
| gate_proj | 3.00× | 2.46× | 1.97× |
| **up_proj** | **3.33×** | **2.75×** | **2.19×** |
| down_proj | 2.14× | 1.72× | 1.55× |

**up_proj universally has the highest scale ratio.** This is the SiLU expansion path — it needs the most amplification because SiLU(x) = x * sigmoid(x) compresses small values toward zero.

### 1.5 The 4B Embedding Anomaly

The 4B model is the ONLY one where embeddings are trained:
- 1.7B: flip=0.07%, scale_ratio=1.000 → **frozen**
- 4B: flip=6.56%, scale_ratio=0.990 → **trained**
- 8B: flip=0.06%, scale_ratio=1.000 → **frozen**

This suggests the embedding freezing is a POLICY CHOICE, not a mathematical necessity. Prism may have experimented with different embedding strategies, and the 4B used a different configuration. Alternatively, the 4B model's smaller hidden dimension (2560 vs 4096) may require embedding adaptation.

---

## 2. Scale Formula Reverse-Engineering

### 2.1 The Embedding Formula

For embeddings, the formula is EXACT:
```
amplitude = mean(|w|)  with R² = 0.9999
```
Embeddings are PTQ-only: `sign(w) * mean(|w|)` per g128. No QAT modification.

### 2.2 The Trunk Formula

For trunk tensors, no single statistic predicts the amplitude well:

| Statistic | R² (gate_proj L17) | R² (v_proj L17) | R² (q_proj L17) |
|---|---|---|---|
| mean\|w\| | 0.184 | 0.384 | 0.648 |
| std | 0.204 | 0.406 | 0.655 |
| L2/sqrt(n) | 0.204 | 0.406 | 0.655 |
| max\|w\| | 0.107 | 0.121 | 0.507 |
| p90 | 0.177 | 0.354 | 0.639 |

The best single predictor is **L2/sqrt(n)** (equivalent to std), but R² is only 0.18-0.65. The Bonsai amplitude is NOT a simple function of any single base weight statistic.

### 2.3 The Ratio Analysis

| Tensor | Best ratio (lowest CV) | Ratio mean | Ratio CV |
|---|---|---|---|
| gate_proj L17 | L2/sqrt(n) | 1.506 | 0.152 |
| v_proj L17 | L2/sqrt(n) | 1.759 | 0.085 |
| q_proj L17 | L2/sqrt(n) | 1.209 | 0.162 |
| **embed** | **mean\|w\|** | **1.000** | **0.002** |

**Key insight**: The Bonsai amplitude is approximately `k * std(w)` where k varies by tensor type (1.2-1.8×). The coefficient of variation is 8-16% for trunk tensors, meaning the scale is NOT purely a function of the base weights — it was TRAINED.

### 2.4 Cross-Tensor Scale Correlations

Within layer 17:
- q_proj ↔ k_proj: **+0.189** (positive — Q and K scales co-vary)
- q_proj ↔ v_proj: +0.078 (weak)
- k_proj ↔ v_proj: -0.111 (slightly negative)
- gate_proj ↔ up_proj: +0.052 (weak — MLP expansion pair is independent)
- gate_proj ↔ down_proj: +0.003 (independent)

**Q and K scales correlate** because they both project into the attention mechanism and must maintain compatible dot-product scales. V, O, and MLP scales are independent.

---

## 3. Optimizer Signature Analysis

### 3.1 Singular Value Distribution

| Layer | Tensor | Base entropy | Bonsai entropy | Base Gini | Bonsai Gini | Base top10% | Bonsai top10% |
|---|---|---|---|---|---|---|---|
| 0 | gate_proj | 8.1 | 8.2 | 0.373 | **0.212** | 0.248 | **0.175** |
| 17 | gate_proj | 8.2 | 8.3 | 0.234 | **0.198** | 0.188 | **0.165** |
| 35 | gate_proj | 8.3 | 8.3 | 0.195 | 0.207 | 0.168 | 0.167 |
| 17 | q_proj | 7.9 | 8.0 | 0.472 | **0.403** | 0.307 | **0.249** |

**Bonsai weights have MORE uniform singular value distributions** (lower Gini, lower top-10% concentration, higher participation ratio). This is consistent with:
- **Muon (Newton-Schulz) optimizer**: orthogonalizes gradients → produces updates with balanced singular values → drives weight matrices toward balanced spectra
- The effect is strongest in early/middle layers, weaker in the last layer

### 3.2 Update Direction Analysis

| Layer | Tensor | cos(W, ΔW) | ΔW entropy | ΔW participation ratio |
|---|---|---|---|---|
| 0 | gate_proj | **-0.269** | 8.3 | 3724 |
| 0 | v_proj | -0.050 | 6.9 | 954 |
| 17 | gate_proj | **-0.178** | 8.3 | 3715 |
| 17 | v_proj | +0.072 | 6.9 | 950 |
| 35 | gate_proj | +0.009 | 8.3 | 3642 |
| 35 | v_proj | +0.028 | 6.9 | 930 |

**The update ΔW has very high participation ratio (~3700 for gate_proj, ~950 for v_proj)**, close to the maximum possible. This means the update has nearly uniform singular values — a strong Muon signature.

**cos(W, ΔW) is NEGATIVE for early layers** (-0.27 for L0 gate_proj), meaning the update partially *reduces* the weight magnitude. This is expected: binarization requires moving from a continuous distribution to {+s, -s}, which involves both sign changes and magnitude changes.

### 3.3 Conclusion: Muon or Similar Orthogonalized Optimizer

The evidence strongly suggests **Muon (Newton-Schulz) or a similar gradient-orthogonalizing optimizer** was used:
1. Bonsai weights have more uniform singular value distributions than base
2. The update ΔW has near-uniform singular values (high participation ratio)
3. The update is partially anti-aligned with W (cosine negative for early layers)
4. The Gini coefficient decreases (more egalitarian singular value distribution)

This matches the Muon optimizer's behavior: `update = lr * Newton-Schulz(grad)`, where Newton-Schulz orthogonalizes the gradient matrix, producing updates with balanced singular values.

---

## 4. Code Spatial Structure

### 4.1 Correlation Analysis

| Tensor | Adjacent row corr | Adjacent col corr | Adjacent group corr | Lag-1 autocorr | FFT peak/mean |
|---|---|---|---|---|---|
| gate_proj L17 | +0.0001 | -0.003 | -0.0003 | -0.018 | 3.58 |
| v_proj L17 | +0.002 | +0.000 | +0.000 | -0.005 | 3.80 |
| q_proj L17 | +0.003 | +0.007 | -0.002 | -0.013 | 3.87 |
| embed | +0.048 | -0.003 | +0.001 | +0.001 | 7.88 |

**Binary codes are essentially spatially white (uncorrelated):**
- Adjacent row correlation: ~0.000 (no row-to-row structure)
- Adjacent column correlation: ~0.000 (no column-to-column structure)
- Adjacent group correlation: ~0.000 (no group-to-group structure)
- Lag-1 autocorrelation: ~0.000 (no sequential structure)
- FFT peak/mean ratio: 3.6-7.9 (close to the theoretical ~3.5 for white noise)

**The only structure is in embeddings** (row correlation = 0.048, FFT peak = 7.88), which reflects the semantic clustering of tokens. Adjacent token IDs in the vocabulary have similar embeddings (because they're often related tokens added in the same vocab extension).

### 4.2 No Structured Patterns

There are NO checkerboard patterns, NO periodic structures, NO spatial correlations. The binary codes are as random as a fair coin flip, conditioned on the base weight signs. This is exactly what STE training produces: each weight independently finds its optimal sign based on the gradient signal, with no spatial coordination.

---

## 5. Training Data Signature

### 5.1 Embedding Norm Distribution

| Model | Base norm mean | Bonsai norm mean | Norm reduction | Peaks |
|---|---|---|---|---|
| 1.7B | 1.541 | 1.233 | 20.0% | 2 peaks |
| 4B | 1.099 | 0.869 | 20.9% | 3 peaks |
| 8B | 1.377 | 1.101 | 20.1% | 1 peak |

**Embedding norms shrink by ~20% after binarization** (consistent across all sizes). This is the natural consequence of binary quantization: `||sign(w) * mean(|w|)|| < ||w||` because binary removes the variance in magnitudes.

### 5.2 Vocab Pruning

All 267 pruned tokens (IDs 151669-151935) decode to empty strings — they are **unused special tokens** that were never trained in the base model. Pruning them saves 267 × hidden_size × 2 (embed + lm_head) params with zero quality loss. This is a trivial optimization, not a methodological insight.

The remaining token 151668 (`<|im_end|>`) is the chat template end marker — critical for instruction following. The pruning boundary is exactly at the last "meaningful" special token.

---

## 6. Training Step Estimation

### 6.1 Flip Rate Decay Model

The flip rate follows an exponential decay with weight magnitude:

```
flip_rate(|w|) = 0.537 * exp(-40.4 * |w|)
```

- At |w|→0: flip_rate = 53.7% (nearly random, slightly above 50%)
- Half-life: |w| = 0.017 (weights above 0.017 have <27% flip rate)
- At |w| = 0.05: flip_rate = 7.1% (large weights are nearly locked)
- At |w| = 0.07: flip_rate = 2.6% (very large weights are frozen)

### 6.2 Interpretation

The exponential decay with decay constant b=40.4 is consistent with STE training where:
- The gradient signal for sign flipping is proportional to the weight magnitude
- Small weights have weak restoring force → high flip probability
- Large weights have strong restoring force → low flip probability
- The decay constant b ≈ lr × T / σ² (where T = training steps, σ = gradient noise)

The fact that flip_rate at |w|→0 is 53.7% (not exactly 50%) suggests a slight asymmetry — possibly from the proximal pull or from the distillation loss providing a weak directional bias even for near-zero weights.

---

## 7. Update Magnitude Per Layer

### 7.1 Weight Delta by Layer (8B)

| Layer | Avg Δ/||W|| | v_proj Δ | gate_proj Δ | up_proj Δ | down_proj Δ |
|---|---|---|---|---|---|
| 0 | 0.983 | 1.111 | 1.060 | 1.101 | 0.793 |
| 8 | 1.045 | 1.147 | 1.101 | 1.190 | 0.885 |
| 12 | 1.126 | 1.279 | 1.178 | 1.315 | 0.941 |
| 17 | 1.203 | 1.366 | 1.273 | 1.410 | 0.995 |
| 22 | **1.304** | **1.689** | 1.410 | 1.482 | 1.031 |
| 24 | **1.312** | 1.692 | 1.371 | 1.466 | 1.026 |
| 26 | 1.306 | 1.699 | 1.363 | 1.446 | 1.011 |
| 30 | 1.280 | 1.621 | 1.388 | 1.390 | 0.992 |
| 35 | 1.146 | 1.398 | 1.485 | 1.232 | 0.968 |

**Key observations:**
- Update magnitude peaks at layers 22-26 (the "exploration zone")
- v_proj moves the MOST (up to 1.70× its base norm at L22-26)
- down_proj moves the LEAST (~0.97× — it's the compression layer, less to change)
- First and last layers are more conservative (~0.98 and ~1.15)
- The pattern is U-shaped: conservative at boundaries, aggressive in the middle

### 7.2 Learning Rate Allocation

The uniform Δ/||W|| ≈ 1.0 across all layers suggests **no layer-wise learning rate scaling** was used. The model was trained with a single global learning rate, and the differences in update magnitude come from the gradient signal (which is naturally larger for middle layers that process richer representations).

---

## 8. Whitepaper Revelations

### 8.1 Official Claims (March 31, 2026)

The Bonsai-8B whitepaper confirms:
- **Format**: Q1_0_g128 (1 bit + FP16 scale per 128 weights) = 1.125 bpw
- **MLX format**: `s_mlx = 2 * s_g`, `b_mlx = -s_g` → 1.25 bpw (MLX overhead)
- **Coverage**: embeddings, attention, MLP, LM head — ALL 1-bit
- **Benchmarks**: 70.5 avg across 6 categories (vs Qwen3-8B at 79.3 — 89% retention)
- **GSM8K**: 88 (vs Qwen3-8B at 93 — 95% retention!)
- **Speed**: 6.2× on RTX 4090, 8.4× on M4 Pro, 44 tok/s on iPhone 17 Pro
- **Energy**: 4-6× lower per token

### 8.2 Method Disclosure

The whitepaper explicitly states the method is **"proprietary Caltech intellectual property"** based on **"mathematically grounded advances"** and **"rigorous mathematics rather than ad hoc heuristics."** It does NOT disclose:
- The training method (STE, optimizer, loss)
- The training data
- The training duration
- The number of steps
- The learning rate

References [5] and [6] are the BitNet papers (Microsoft's 1-bit LLM work), confirming the connection to the BitNet lineage. The "Caltech IP" likely refers to the optimal-scale theory from Hassibi's group.

### 8.3 GGUF vs MLX Format

| Format | bpw | Size | Compression |
|---|---|---|---|
| GGUF Q1_0 | 1.125 | 1.15 GB | 14.2× |
| MLX 1-bit | 1.25 | 1.28 GB | 12.8× |

GGUF is more efficient because it stores only the scale (1 FP16 per 128 weights). MLX requires both scale and bias (2 FP16 per 128 weights), adding 0.125 bpw overhead. The whitepaper notes this is a "current limitation of MLX" that will be resolved when MLX supports scale-only formats.

---

## 9. Summary of ALL Secrets (Extended)

### Structural Secrets
1. TRUE 1-bit binary: every weight is +s or -s, no zeros, no ternary
2. Perfect symmetry: bias = -scale/2 exactly, |min| = |max| exactly
3. Group size 128, per-row, per-group-of-128
4. All 2D tensors quantized (including embed + lm_head)
5. Only 1D RMSNorms kept FP16 (0.004% of params)
6. 267 unused special tokens pruned from vocab
7. MLX: LSB-first uint32 packing, w = scale * bit + bias
8. GGUF: 1.125 bpw (scale only), MLX: 1.25 bpw (scale + bias)

### Training Secrets (Reverse-Engineered)
9. PTQ init = sign(w) * mean(|w|) (simplest possible binary PTQ)
10. No Hadamard rotation, no GPTQ, no sophisticated initialization
11. 22-30% sign flip rate (model-specific, smaller models flip more)
12. Flip rate = 0.537 * exp(-40.4 * |w|) — magnitude-dependent
13. Scale growth = compensation: 1.0-3.6× (tensor-type dependent)
14. No FP16 2D islands — compensation via scale growth only
15. Layer 0 input norm compensates for frozen embeddings (22.6% change)
16. Embeddings frozen in 1.7B/8B, trained in 4B (policy choice)
17. Perfect 50/50 code balance per group (implicit regularization)
18. No spatial correlations in codes (white noise structure)
19. Middle layers (22-26) explore most (highest delta, highest flips)
20. v_proj universally most stable, gate_proj/up_proj most aggressive

### Optimizer Secrets (Reverse-Engineered)
21. **Muon (Newton-Schulz) optimizer** — evidenced by:
    - Bonsai singular values more uniform than base (Gini decreases)
    - Update ΔW has near-uniform singular values (PR ~3700)
    - Update partially anti-aligned with W (cosine negative for early layers)
22. No layer-wise LR scaling (uniform Δ/||W|| ≈ 1.0)
23. up_proj universally has highest scale ratio (SiLU compensation)

### Cross-Model Secrets
24. Universal patterns: v_proj most stable, up_proj most amplified, perfect balance
25. Smaller models compensate harder (higher flips, higher scale ratios)
26. Q/K scale correlation positive (+0.19), V/O/MLP scales independent
27. Binary better conditioned than ternary (condition 7-9 vs 158-634M)
28. 20% embedding norm reduction is universal (consequence of binarization)

### Whitepaper Secrets
29. "Proprietary Caltech IP" — method NOT disclosed
30. BitNet lineage (references [5,6] = BitNet b1.58 and BitNet)
31. 89% quality retention at 14.2× compression (70.5 vs 79.3 avg score)
32. 95% math retention (GSM8K: 88 vs 93)
33. 44 tok/s on iPhone, 6.2× on RTX 4090, 5.6× energy reduction

---

## 10. The Inferred Complete Recipe

```
Bonsai 1-Bit Training Recipe (Reverse-Engineered):

1. PTQ INITIALIZATION
   - Method: sign(w) * mean(|w|), group_size=128
   - Applied to: ALL 2D tensors (embed, lm_head, attention, MLP)
   - NOT applied: 1D RMSNorm weights (kept FP16)
   - Prune unused vocab tokens (267 tokens)

2. QAT TRAINING
   - Quantization: Straight-Through Estimator (STE)
   - Optimizer: Muon (Newton-Schulz gradient orthogonalization)
   - Scale: trainable per-group, free growth
   - No sign locking (22-30% flip rate)
   - No FP16 2D islands
   - No Hadamard rotation
   - No GPTQ
   - Loss: likely self-distillation (KL on logits) + codebook loss
   - Embeddings: frozen or very low LR (except 4B where trained)
   - LM head: low LR (10% flip rate vs 22% trunk)

3. PACKING
   - GGUF Q1_0: 1 bit per weight + 1 FP16 scale per 128 weights = 1.125 bpw
   - MLX: 1 bit per weight + 2 FP16 (scale+bias) per 128 = 1.25 bpw
   - LSB-first bit packing
   - Symmetric: w = {-scale, +scale} (GGUF) or w = scale*bit + bias (MLX)

4. DEPLOYMENT
   - Custom inline dequantization kernels (no FP16 materialization)
   - Cross-platform: CUDA, Metal, CPU, mobile
```

---

## 11. What We Still Don't Know

1. **Exact training data**: Likely a general text corpus (not OOD/multimodal — embeddings are frozen)
2. **Exact loss function**: Likely KL distillation + codebook regularization, but the exact form is unknown
3. **Training duration**: The flip rate decay constant (b=40.4) suggests thousands of steps, but exact count unknown
4. **Learning rate**: Unknown, but the uniform update magnitude suggests no warmup or schedule
5. **Batch size**: Unknown
6. **The "Caltech IP"**: The mathematical theory behind the method — likely related to optimal-scale theory and ℓ∞ regularization from Hassibi's group, but the specific formulation is proprietary
7. **Why embeddings are frozen in 1.7B/8B but trained in 4B**: Could be a hyperparameter search artifact or a deliberate policy difference
8. **Whether ternary and binary were trained separately**: Yes (confirmed by separate model releases with different README claims — 14× vs 9× compression)
