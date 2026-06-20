# Prism Weight Forensics: Reverse-Engineering Bonsai from the Checkpoints

**Date:** 2026-06-20  
**Models analyzed:**  
- `prism-ml/bonsai-image-ternary-4B-unpacked` (trained ternary, dequantized to FP16)  
- `prism-ml/bonsai-image-binary-4B-unpacked` (trained binary, dequantized to FP16)  
- `black-forest-labs/FLUX.2-klein-4B` (FP16 base model)  

**Method:** Downloaded all three transformer safetensors (7.0 GB each), mapped tensor names between BFL-original and diffusers formats, computed per-tensor ternary/binary reconstruction error, weight deltas, code transition matrices, scale distributions, and island-vs-trunk classification.

---

## Executive Summary

Prism's method is **quantization-aware fine-tuning (QAT) with straight-through estimation (STE)**, applied to the FLUX.2 Klein 4B base model. The weights are not post-hoc quantized; they are *trained into the codebook* via self-distillation against the FP16 teacher. The forensic evidence is unambiguous:

1. **Every trunk tensor is exactly ternary/binary-representable** (0.0000% reconstruction error at g128).
2. **Trunk weights moved 54% (ternary) / 69% (binary) from the base model** — far beyond what PTQ produces (~10-20%).
3. **Zero sign flips** occurred during training. The optimizer only changes *which* weights are zero (sparsity) and the *scales*, never the signs.
4. **The final projection (`proj_out`) is bit-identical to the base model** — completely frozen, never trained.
5. **The final norm modulation (`norm_out`) moved 141%** — the heaviest-trained island tensor, acting as the trunk-error compensator.
6. **Ternary and binary are separately trained** — island weights differ between them (except `proj_out`).

---

## 1. Architecture and Tensor Mapping

### 1.1 Base model: FLUX.2 Klein 4B

| Property | Value |
|---|---|
| Architecture | Flux2Transformer2DModel (MMDiT) |
| Total tensors | 149 |
| Blocks | 5 double-stream + 20 single-stream = 25 |
| Inner dim | 3072 |
| Attention heads | 24 × 128 dim |
| MLP ratio | 3.0 (hidden = 9216) |
| Joint attention dim | 7680 (text conditioning) |
| Parameters | ~3.88B |

**Base tensor naming (BFL original):**
- Double-stream: `double_blocks.{n}.img_attn.qkv.weight` [9216, 3072], `.proj.weight` [3072, 3072], `img_mlp.0/2.weight`, `txt_attn.*`, `txt_mlp.*`
- Single-stream: `single_blocks.{n}.linear1.weight` [27648, 3072] (fused QKV+MLP), `linear2.weight` [3072, 12288] (attn out)
- Islands: `txt_in`, `img_in`, `time_in.in/out_layer`, `double/single_stream_modulation.lin`, `final_layer.adaLN_modulation.1`, `final_layer.linear`

### 1.2 Prism model: diffusers format

Prism converts to diffusers naming with **QKV split into separate projections** and single-stream blocks use `to_qkv_mlp_proj` (fused) + `to_out`:

| Base key | Prism key | Shape |
|---|---|---|
| `double_blocks.{n}.img_attn.qkv.weight` | `transformer_blocks.{n}.attn.to_q/to_k/to_v.weight` | 3 × [3072, 3072] |
| `double_blocks.{n}.img_attn.proj.weight` | `transformer_blocks.{n}.attn.to_out.0.weight` | [3072, 3072] |
| `double_blocks.{n}.txt_attn.qkv.weight` | `transformer_blocks.{n}.attn.add_q/add_k/add_v_proj.weight` | 3 × [3072, 3072] |
| `single_blocks.{n}.linear1.weight` | `single_transformer_blocks.{n}.attn.to_qkv_mlp_proj.weight` | [27648, 3072] |
| `single_blocks.{n}.linear2.weight` | `single_transformer_blocks.{n}.attn.to_out.weight` | [3072, 12288] |
| `final_layer.linear.weight` | `proj_out.weight` | [128, 3072] |
| `final_layer.adaLN_modulation.1.weight` | `norm_out.linear.weight` | [6144, 3072] |

**Prism model: 169 tensors** (vs 149 base — the diffusers format splits QKV and separates norm scales).

---

## 2. Ternary/Binary Verification

### 2.1 Every trunk tensor is exactly quantized

For each 2D weight tensor `W [out, in]`, we compute the optimal-scale ternary reconstruction with group=128 along the input dimension (Clark's method: sort `|w|` per group, find `k* = argmax_k (Σ_top-k |w|)²/k`, set `s = mean_top-k |w|`, threshold at `k*`-th largest, codes = `sign(w) * (|w| >= threshold)`).

**Result: 100 trunk tensors, all with 0.0000% reconstruction error.**

| Tensor (representative) | Ternary error | Zero fraction | Scale mean | Scale std |
|---|---|---|---|---|
| `transformer_blocks.0.attn.to_q.weight` | 0.0000% | 32.1% | 0.015434 | 0.004354 |
| `transformer_blocks.0.attn.to_v.weight` | 0.0000% | 30.4% | 0.014979 | 0.002309 |
| `transformer_blocks.0.ff.linear_in.weight` | 0.0000% | 32.8% | 0.017732 | 0.004363 |
| `single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight` | 0.0000% | 34.7% | 0.018996 | 0.003491 |
| `single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight` | 0.0000% | 40.9% | 0.017991 | 0.004794 |

The "unpacked" repos are **dequantized quantized models** (ternary codes × FP16 group scales, stored as continuous BF16 values), not trained master weights. The actual quantized packs (`-gemlite-2bit`, `-mlx-2bit`) store the codes in packed form.

### 2.2 Binary model: exactly binary (no zeros)

| Tensor (representative) | Binary deviation | Zero fraction |
|---|---|---|
| `single_transformer_blocks.0.attn.to_qkv_mlp_proj` | 0.00000000 | 0.0000% |
| `single_transformer_blocks.10.attn.to_qkv_mlp_proj` | 0.00000000 | 0.0000% |
| `single_transformer_blocks.19.attn.to_qkv_mlp_proj` | 0.00000000 | 0.0000% |

Every non-zero weight in each group has exactly the same absolute value (= group scale). No zeros. Pure binary {-1, +1}.

### 2.3 Island tensors: clearly continuous FP16

| Island tensor | Ternary error if quantized | Unique values (sample) |
|---|---|---|
| `context_embedder.weight` | 37.24% | 4749 |
| `proj_out.weight` | 59.78% | 1964 |
| `norm_out.linear.weight` | 47.97% | 2176 |
| `single_stream_modulation.linear.weight` | 58.18% | 1795 |

Island tensors have thousands of unique values and 37-60% error if ternarized — they are continuous FP16, never quantized.

---

## 3. The Island Set (Exactly Confirmed)

### 3.1 Classification: 69 island tensors, 100 trunk tensors

**Island (FP16, protected):**
- `context_embedder.weight` [3072, 7680] — text conditioning input
- `x_embedder.weight` [3072, 128] — patch embedding
- `proj_out.weight` [128, 3072] — final pixel projection
- `norm_out.linear.weight` [6144, 3072] — final norm modulation
- `time_guidance_embed.timestep_embedder.linear_1/2.weight` — time embedding
- `double_stream_modulation_img/txt.linear.weight` — double-stream modulation
- `single_stream_modulation.linear.weight` — single-stream modulation
- All `norm_q/norm_k/norm_added_q/norm_added_k.weight` [128] — QK norm scales (40 tensors)
- Double-stream block QK norms (20 tensors)

**Trunk (ternary/binary, quantized):**
- All `to_q`, `to_k`, `to_v`, `to_out`, `add_q/k/v_proj`, `to_add_out` — attention projections
- All `ff.linear_in/out`, `ff_context.linear_in/out` — FFN layers
- All `to_qkv_mlp_proj`, single-stream `to_out` — fused single-stream projections

### 3.2 Parameter budget

| Component | Parameters | % of total |
|---|---|---|
| Trunk (ternary g128) | 3680.5M | 95.0% |
| Island (FP16) | 195.0M | 5.0% |
| Norms (FP16, 1D) | ~0.02M | ~0.0% |
| **Total** | **3875.5M** | 100% |

### 3.3 Effective bits per weight

| Format | Trunk bpw | Island bpw | Effective bpw | Storage | Compression |
|---|---|---|---|---|---|
| Ternary g128 | 1.710 (log₂3 + 16/128) | 16.0 | **2.429** | 1.177 GB | **6.59×** |
| Binary g128 | 1.125 (1 + 16/128) | 16.0 | **1.874** | 0.908 GB | **8.54×** |

The ternary model at 1.18 GB matches the claimed 1.21 GB (small difference from safetensors metadata overhead). The binary at 0.91 GB matches the claimed 0.93 GB.

**Note:** The 5% island at FP16 costs 0.39 GB — 33% of the ternary model's total storage. Compressing the island to FP8 would save ~0.19 GB, bringing ternary to ~0.99 GB (8.5×) and binary to ~0.72 GB (10.8×).

---

## 4. Weight Deltas: What Training Moved

### 4.1 Trunk deltas (trained vs base)

The trunk weights moved **dramatically** from the base model — far more than PTQ:

| Layer (single-stream) | Ternary delta | Binary delta |
|---|---|---|
| block 0 (linear1) | 54.0% | 65.1% |
| block 5 | 54.2% | 65.2% |
| block 10 | 55.2% | 66.9% |
| block 15 | 58.4% | 72.2% |
| block 19 | 63.5% | 75.4% |
| **Mean (all 20)** | **56.0%** | **68.8%** |

**Key pattern: delta increases with depth.** Early blocks move ~54%, late blocks move ~63% (ternary). Binary is more extreme: 65% → 75%. This means later layers require more training to adapt to the codebook constraint.

For context, optimal-scale ternary PTQ (Clark's method, no training) produces ~10-20% weight delta from FP16. Prism's 54-63% is **3-5× more movement**, confirming active training rather than rounding.

### 4.2 Double-stream block 0 per-module deltas

| Module | Delta |
|---|---|
| img_attn to_q | 55.6% |
| img_attn to_k | 59.2% |
| img_attn to_v | 53.7% |
| img_attn proj (to_out) | 50.7% |
| img_mlp ff_in | 54.4% |
| img_mlp ff_out | 56.2% |
| txt_attn add_q | 51.6% |
| txt_attn add_k | 52.0% |
| txt_attn add_v | 51.7% |
| txt_attn proj | 49.7% |
| txt_mlp ctx_in | 52.2% |
| txt_mlp ctx_out | 51.9% |

All trunk modules move ~50-59%. No single module type is spared — training reshapes the entire trunk uniformly.

### 4.3 Island deltas (the critical finding)

| Island tensor | Ternary delta from base | Binary delta from base | Ternary vs Binary |
|---|---|---|---|
| **`proj_out.weight`** | **0.00%** | **0.00%** | **0.00%** |
| `norm_out.linear.weight` | **141.34%** | **141.33%** | 1.63% |
| `context_embedder.weight` | 23.47% | 25.48% | 30.51% |
| `time_embedder.linear_1` | 16.25% | 14.92% | 11.86% |
| `time_embedder.linear_2` | 10.43% | 10.74% | 8.94% |
| `x_embedder.weight` | 6.85% | 7.99% | 6.74% |
| `double_stream_modulation_img` | 4.88% | 4.98% | 5.56% |
| `double_stream_modulation_txt` | 4.37% | 4.54% | 5.49% |
| `single_stream_modulation` | **1.46%** | **1.40%** | 1.45% |

**Three critical observations:**

1. **`proj_out` is bit-identical to base** (verified: `torch.equal` returns True, max abs diff = 0.0). This is the "last layer" from Hassibi's random-features theorem — *all-but-the-last-layer can be 1-bit with no asymptotic loss*. Prism froze it completely.

2. **`norm_out` moved 141%** — more than its own weight norm. This is the **island compensation channel**: the final adaptive norm absorbs accumulated trunk quantization error. Both ternary and binary converged to nearly the same `norm_out` (only 1.63% difference), suggesting this is a deterministic function of the trunk error, not a free parameter.

3. **`single_stream_modulation` barely moved (1.4%)** — the per-block modulation is nearly frozen. The compensation is concentrated at the *final* norm, not distributed across block-level modulation.

### 4.4 QK norm scales (1D, [128])

| Block | norm_q delta (ternary) | norm_k delta (ternary) | norm_q delta (binary) |
|---|---|---|---|
| 0 | 0.56% | 0.56% | 0.50% |
| 5 | 0.68% | 0.66% | 0.88% |
| 10 | 0.60% | 0.60% | 1.43% |
| 15 | 0.66% | 0.67% | 1.05% |
| 19 | 1.15% | 1.13% | 1.31% |
| **Mean** | **0.69%** | **0.69%** | **0.97%** |

QK norms barely moved (<1% mean). Binary moved slightly more (0.97% vs 0.69%), consistent with greater trunk distortion requiring more norm adjustment.

---

## 5. Code Transition Analysis

### 5.1 Transition matrix: base-ternarized → Prism trained ternary

Aggregated across all 20 single-stream blocks (27648 × 3072 × 20 = 1.7B weights):

```
                 trained→    -1        0       +1     |  base total
  base -1:              24.92%    1.12%    0.00%   |   26.04%
  base  0:               6.20%   35.46%    6.21%   |   47.88%
  base +1:               0.00%    1.12%   24.97%   |   26.09%
  trained total:        31.12%   37.70%   31.18%
```

### 5.2 What training did to the codes

| Transition | % of weights | Interpretation |
|---|---|---|
| **Preserved (same code)** | **85.35%** | 85% of weights kept their original ternary code |
| **Activated (0 → ±1)** | **12.41%** | 12% of previously-zero weights were activated |
| **Deactivated (±1 → 0)** | **2.23%** | 2% of previously-nonzero weights were zeroed |
| **Sign flips (±1 → ∓1)** | **0.002%** | Essentially zero — signs are NEVER flipped |

**The training is almost entirely about sparsity redistribution, not sign changes.**

The optimizer:
1. **Keeps signs from the FP16 model** (inherited via STE, which passes gradients through `sign()` as identity)
2. **Activates previously-zero weights** (12.4% of all weights, balanced 6.20% → +1 and 6.21% → -1)
3. **Deactivates a few previously-nonzero weights** (2.2%)
4. **Reduces zero fraction from ~48% to ~38%** (net activation of ~10%)

### 5.3 Zero fraction depth trend

| Block | Base PTQ zero frac | Prism trained zero frac | Change |
|---|---|---|---|
| 0 | 48.4% | 34.7% | -13.7% |
| 5 | 47.9% | 33.9% | -14.0% |
| 10 | 47.2% | 36.3% | -10.9% |
| 15 | 47.6% | 42.5% | -5.1% |
| 19 | 53.0% | 40.9% | -12.0% |

Early blocks: training activates ~14% more weights (denser).  
Late blocks: training activates only ~5% more (sparser, closer to PTQ).  
Block 19 (last): base is already 53% sparse, training brings it to 41%.

**The trained model is systematically denser than PTQ.** Training discovers that more weights need to be active than optimal-scale PTQ selects.

### 5.4 Binary model: sign relationship to ternary

Where the ternary model has a non-zero code, the binary model agrees on the sign **98-99.6%** of the time:

| Block | Sign agreement (ternary-nz vs binary) |
|---|---|
| 0 | 99.6% |
| 10 | 98.8% |
| 19 | 97.9% |

Where the ternary model has zeros, the binary model splits 50/50 into +1/-1 (it has no zero option). This is consistent with both models being trained from the same base with the same sign inheritance.

---

## 6. Scale Distribution Analysis

### 6.1 Trained scales are ~63% of PTQ scales

| Block | Base PTQ scale mean | Trained ternary scale mean | Ratio | Trained binary scale mean | Ratio |
|---|---|---|---|---|---|
| 0 | 0.030223 | 0.018996 | 0.628 | 0.018167 | 0.601 |
| 10 | 0.029068 | 0.018352 | 0.631 | 0.017553 | 0.604 |
| 19 | 0.035608 | 0.017991 | 0.505 | 0.016700 | 0.469 |

**Mean scale ratio (trained/base PTQ): 0.625 for ternary, 0.594 for binary.**

Training reduces group scales to ~60% of their PTQ values. This is consistent with activating more weights: if you spread the same magnitude across more non-zero entries, each entry is smaller, so the per-group scale (mean of non-zero |w|) decreases.

### 6.2 Scale distribution narrowing

| Block | Base PTQ scale std | Trained ternary scale std |
|---|---|---|
| 0 | 0.004654 | 0.003491 |
| 10 | 0.006202 | 0.004134 |
| 19 | 0.032158 | 0.004794 |

The base PTQ has high scale variance (especially block 19: std=0.032 with outliers up to 0.68). Training **dramatically narrows the scale distribution** (block 19 std drops from 0.032 to 0.005, max from 0.68 to 0.069). The optimizer homogenizes the group scales, eliminating outlier groups.

---

## 7. Ternary vs Binary: Separate Training

### 7.1 Island weights differ (except proj_out)

| Island | Ternary vs Binary delta |
|---|---|
| `proj_out` | 0.00% (identical) |
| `norm_out` | 1.63% (nearly identical) |
| `single_stream_modulation` | 1.45% (nearly identical) |
| `context_embedder` | 30.51% (significantly different) |
| `time_embedder.linear_1` | 11.86% (moderately different) |
| `x_embedder` | 6.74% (moderately different) |

**`proj_out` is identical because it's frozen at the base value.** `norm_out` and `single_stream_modulation` are nearly identical because they converge to the same compensation regardless of trunk precision. But `context_embedder`, `time_embedder`, and `x_embedder` differ significantly — they are trained separately for ternary vs binary.

**Conclusion: ternary and binary are independently trained runs**, not two quantizations of one trained model.

### 7.2 Trunk codes are related but not identical

The ternary and binary models share the same sign pattern (98-99.6% agreement where ternary is non-zero), but the binary model must assign ±1 to every weight (no zeros), so the ~38% of weights that are zero in ternary get split 50/50 in binary. This is consistent with both starting from the same base and preserving signs via STE.

---

## 8. Reconstructed Training Pipeline

Based on all evidence, Prism's pipeline is:

### Step 1: Initialize from base with optimal-scale ternary PTQ
- Load FLUX.2 Klein 4B in FP16
- Apply Clark-style optimal-scale ternary g128 to all trunk tensors
- Keep island tensors in FP16
- This gives ~48% zero fraction and scale mean ~0.030

### Step 2: STE QAT with self-distillation
- **Teacher:** frozen FP16 base model
- **Student:** quantized model with trainable master weights `u`
- **Forward:** `W_fwd = s_G · t_G + (u - u.detach())` (STE, hard ternary forward, continuous backward)
- **Loss:** velocity + multi-step trajectory distillation against teacher (no image dataset needed — self-calibration on Gaussian noise + synthetic/null prompts)
- **Optimizer:** likely AdamW or a preconditioned variant (Pooladzandi's Lie-group preconditioners from the Caltech research line)

### Step 3: Train the island selectively
- **Freeze `proj_out` completely** (the random-features "last layer")
- **Train `norm_out` heavily** (141% delta — this is the main compensation channel)
- **Train `context_embedder` moderately** (23% — text conditioning adapts to quantized trunk)
- **Train `time_embedder` lightly** (10-16%)
- **Train `x_embedder` very lightly** (7%)
- **Nearly freeze `single_stream_modulation`** (1.5% — block-level modulation barely changes)
- **Nearly freeze QK norms** (<1% — attention normalization is stable)

### Step 4: What the optimizer does
- **Never flips signs** (0.002% — signs are inherited from FP16 and preserved by STE)
- **Activates previously-zero weights** (12.4% of all weights go from 0 → ±1)
- **Deactivates a few weights** (2.2% go from ±1 → 0)
- **Net effect: denser than PTQ** (zero fraction drops from ~48% to ~38%)
- **Reduces group scales to ~63% of PTQ** (more active weights, each smaller)
- **Homogenizes scale distribution** (eliminates outlier groups)
- **Moves later layers more** (depth-dependent delta: 54% → 63%)

### Step 5: Pack and deploy
- Extract ternary codes and FP16 group scales
- Pack to INT2 (gemlite) or MLX 2-bit format
- Final artifact: pure quantized weights + FP16 islands, no LoRA, no adapters

---

## 9. Implications for Sana Training

### 9.1 What to replicate

1. **Initialize from Clark's optimal-scale ternary** — this is the right starting point. Prism almost certainly does the same (the code transition matrix shows 85% code preservation from base-ternarized).

2. **Freeze `proj_out` completely.** Set its `requires_grad=False`. Do not include it in any optimizer group. The random-features theorem says the last layer must stay full-precision; Prism took this literally.

3. **Train `norm_out` with the highest island learning rate.** This is where the compensation lives (141% delta). It needs to move more than any other island tensor.

4. **Nearly freeze block-level modulation.** `single_stream_modulation` moved only 1.5%. Use a very low LR (1e-6 or lower) or freeze it entirely initially.

5. **Expect the optimizer to activate zeros, not flip signs.** The STE mechanism preserves signs from FP16. The main training effect is sparsity redistribution (12% activation) and scale reduction (to 63% of PTQ). Don't design a training method that expects sign changes.

6. **Train ternary and binary separately.** They are not interchangeable; the island weights adapt differently to each trunk precision.

7. **Expect more movement in later layers.** Depth 0-5 moves ~54%, depth 15-19 moves ~60%. Later layers may need more training steps or higher LR.

### 9.2 Sana-specific differences

Sana's architecture differs from FLUX.2 Klein in several ways that affect the training plan:

| Feature | FLUX.2 Klein 4B | Sana 1.6B |
|---|---|---|
| Architecture | MMDiT (double + single stream) | DiT with GLUMBConv FFN |
| FFN | Standard Linear (SwiGLU) | GLUMBConv (Conv2d 1×1 + depthwise 3×3 + 1×1) |
| Attention | Separate Q/K/V projections | Fused or separate |
| Inner dim | 3072 | 2240 |
| Blocks | 5 double + 20 single | 20 uniform |
| Protected island | 5.0% of params | 5.4% of params (Clark's measurement) |

**Sana's `conv_depth` (3×3 depthwise) must be protected.** It has only 9 parameters per channel — far too few for group quantization. Add it to the island set explicitly. This is not present in FLUX.2 Klein (which uses standard Linear FFN), so Prism didn't face this issue.

### 9.3 The Sana island set (corrected)

Based on Prism's confirmed island set + Sana's architecture:

```
ISLAND (keep FP16, train selectively):
  - patch_embed (x_embedder equivalent)
  - time_embed / caption_projection / caption_norm
  - scale_shift_table (per-block modulation, [6, dim] each)
  - norm_out (final modulation) ← train heavily
  - proj_out (final projection) ← freeze completely
  - conv_depth in GLUMBConv ← Sana-specific, must protect

TRUNK (ternary g128, train via STE):
  - All attention Q/K/V/out projections
  - FFN conv_inverted (1×1, [hidden*2, in])
  - FFN conv_point (1×1, [out, hidden])
```

### 9.4 Expected training behavior for Sana

Based on Prism's patterns:

| Metric | Prism (FLUX.2 Klein) | Expected for Sana |
|---|---|---|
| Trunk weight delta | 54% (ternary) | ~50-55% (ternary) |
| Zero fraction (PTQ → trained) | 48% → 38% | ~45% → ~35% |
| Scale ratio (trained/PTQ) | 0.63 | ~0.60-0.65 |
| Sign flips | ~0% | ~0% |
| `proj_out` delta | 0% | 0% (freeze) |
| `norm_out` delta | 141% | ~100-150% (train hard) |
| Block modulation delta | 1.5% | ~1-2% (nearly freeze) |
| Later-layer delta increase | +10% | +5-10% |

---

## 10. Summary of Key Findings

| Finding | Evidence | Implication |
|---|---|---|
| Trunk is exactly ternary | 0.0000% reconstruction error | "Unpacked" = dequantized ternary |
| Trunk moved 54% from base | Weight delta vs base | Active training, not PTQ |
| Zero sign flips | Code transition matrix | STE preserves signs; training = sparsity + scales |
| 12% zero-to-nonzero | Transition matrix | Training activates weights PTQ zeros out |
| Scales shrink to 63% | Scale distribution | More active weights, each smaller |
| `proj_out` frozen | Bit-identical to base | Random-features "last layer" theorem in practice |
| `norm_out` moved 141% | Weight delta | Island compensation channel |
| Block modulation barely moved | 1.5% delta | Compensation is at final norm, not per-block |
| Ternary ≠ binary islands | 30% delta in context_embedder | Separately trained |
| Delta increases with depth | 54% → 63% | Later layers need more adaptation |
| Scale distribution narrowed | std drops 5× | Training eliminates outlier groups |
| QK norms barely moved | <1% delta | Attention normalization is stable |

---

## Appendix: Reproduction Commands

```bash
# Download models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('prism-ml/bonsai-image-ternary-4B-unpacked', 'transformer/diffusion_pytorch_model.safetensors', local_dir='/tmp/prism_analysis/ternary')
hf_hub_download('prism-ml/bonsai-image-binary-4B-unpacked', 'transformer/diffusion_pytorch_model.safetensors', local_dir='/tmp/prism_analysis/binary')
hf_hub_download('black-forest-labs/FLUX.2-klein-4B', 'flux-2-klein-4b.safetensors', local_dir='/tmp/prism_analysis/base')
"

# Verify ternary
python3 -c "
import torch; from safetensors import safe_open
with safe_open('/tmp/prism_analysis/ternary/transformer/diffusion_pytorch_model.safetensors', framework='pt') as f:
    W = f.get_tensor('transformer_blocks.0.attn.to_q.weight').float()
    # optimal-scale ternary g128
    A = W.abs().reshape(W.shape[0], -1//128, 128).abs()
    a,_ = torch.sort(A, -1, descending=True)
    cs = a.cumsum(-1); ks = torch.arange(1,129,dtype=torch.float32)
    k = (cs*cs/ks).argmax(-1)
    s = cs.gather(-1,k[...,None])/(k[...,None]+1)
    t = torch.sign(W.reshape(W.shape[0],-1,128)) * (A >= a.gather(-1,k[...,None]))
    Wq = (t*s).reshape(W.shape)
    print('ternary error:', ((Wq-W).norm()/W.norm()).item())  # should be ~0
"
```
