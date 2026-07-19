# Ternary Bonsai Deep Forensics: The Recipe, Extracted From the Weights

**Date:** 2026-07-02
**Inputs:** `prism-ml/Ternary-Bonsai-8B-unpacked` vs `Qwen/Qwen3-8B` (full-coverage, all 399 tensors, no sampling), `prism-ml/Ternary-Bonsai-4B-gguf` (Q2_0 byte-level), both Prism whitepapers, Qwen3.5-9B config + tensor index.
**Tools:** `scripts/deep_compare_ternary_full.py` → `.analysis/bonsai_ternary/deep_full_8b.json`; `scripts/analyze_bonsai_ternary_gguf.py` → `q2_0_forensics{,_4b}.json`.

---

## Executive summary — the six load-bearing facts

1. **The quantizer is BitNet b1.58's absmean rule, exactly.** Per group of 128 (input dim): `s = mean(|w|)`, `code = clip(round(w/s), -1, 1)`. Proof: the trained embedding matches absmean+round PTQ of the *base* embedding at **99.999%** code agreement with scale ratio exactly 1.000 (log-std 0.0018). The MSE-optimal sort rule (Clark's default init) does NOT match (85.1% on embed, zeros 47.7% vs observed 37.2%).
2. **Embeddings were never trained.** Pure PTQ of Qwen3-8B `embed_tokens`, frozen. `lm_head` was only lightly trained (85.5% of codes still equal absmean-PTQ of base; sign inheritance 98.5%).
3. **All 1D islands are frozen.** Final `model.norm` is bit-identical to base; q/k norms moved 0.01%; layernorms ≤0.6% on average. Unlike Prism's FLUX image models (norm compensator moved 141%), the LLM recipe trains **only the 2D trunk + lm_head**.
4. **The trunk was genuinely retrained, far and full-rank.** Relative movement 71% (cosine 0.75) — an actual distillation run, not clever rounding. Delta spectra: stable rank 164/4096 (q_proj L17), 469/4096 (gate L21); top-64 SVD directions carry only ~10% of delta energy → **a LoRA-style shortcut cannot reproduce this** (though it's far more structured than noise).
5. **Training changes ~30% of codes from the PTQ init, almost never signs.** absmean-PTQ of base already agrees with final codes at 66–71% (trunk) / 85% (head) / 100% (embed). Changes are dominated by support moves (activation 11.7%, deactivation 16.9% of entries); direct +1↔−1 flips are 1.8%. Among final nonzeros, 91% keep the base weight's sign; the 9% exceptions are mostly near-zero base weights that got promoted.
6. **Sparsity is emergent and structured.** Global zeros 37.2% (8B) / 37.1% (4B). Layer-1 down_proj (8B) ends at 61.7% zeros with **4,287 of 12,288 columns fully zero** — QAT structurally pruned a third of that layer's MLP channels. Per-group support k is tightly concentrated (mean ~75/128, p1=58, p99=88; zero groups: none; saturated groups: none).

The "7.5×/9.36×" arithmetic: weights are exactly ternary, so GGUF Q2_0 g128 (2 bits + FP16 scale/128 = 2.125 b/w) packs them losslessly; 16/2.125 = 7.53×. Whitepaper's 9.36× uses the idealized 1.71 b/w (log₂3 + scale overhead). `PQ2_0` is the same size as Q2_0 (1075.0 MB) — a kernel-layout repack, not extra compression; `Q2_0_g64` = 2.25 b/w. Quality price at 8B: 79.3 → 75.5 avg (−3.8 pts); the 4B ternary scores 70.7.

Neither whitepaper discloses training method — both are deployment/benchmark docs. References cite BitNet [2310.11453] and BitNet b1.58 [2402.17764], consistent with everything above.

---

## Full-coverage 8B results (trained vs base)

### Global

| metric | value | reading |
|---|---|---|
| zero_trained | 0.372 | vs absmean-PTQ-of-base 0.320, MSE-PTQ 0.477 |
| agree_absmean | 0.696 | PTQ init already right for ~70% of codes |
| agree_mse | 0.686 | MSE rule no better — and embed kills it |
| sign_inherit | 0.912 | signs come from base |
| topk_overlap | 0.774 | support ≈ top-k(|base|) with 23% substitution |
| ab_activation / ab_deactivation / ab_signflip | 0.117 / 0.169 / 0.018 | support moves, not sign flips |
| rel_movement / cosine | 0.711 / 0.747 | real training, not rounding |

### By kind (selected)

| kind | zeros | agree_absmean | sign_inherit | movement | note |
|---|---|---|---|---|---|
| embed | 0.309 | **1.000** | 1.000 | 0.512 (= pure quant error) | frozen PTQ |
| lm_head | 0.319 | 0.855 | 0.985 | 0.548 | lightly trained |
| down_proj | 0.399 | 0.710 | 0.936 | 0.646 | sparsest, moves least in trunk |
| v_proj | 0.395 | 0.699 | 0.938 | 0.738 | scales grew 1.84× vs base absmean |
| gate_proj | 0.383 | 0.631 | 0.882 | 0.789 | most retrained |
| up_proj | 0.371 | 0.635 | 0.882 | 0.810 | most movement |
| q_proj | 0.368 | 0.616 | 0.862 | 0.742 | lowest PTQ agreement |

### By layer

U-shaped: L0 and L35 stay closest to PTQ (agree 0.67–0.70, movement ~0.70–0.72); mid-network L19–27 is most retrained (movement 0.78, agree 0.64). Zero fraction: 0.345 (L0) → 0.40–0.42 (L1–3 and L19–22) → 0.346 (L35). The 4B shows the same signature (peak 0.394 at L19–21; `blk.2.ffn_down` 54.9% zeros).

### Islands (1D)

| tensor family | mean movement |
|---|---|
| input_layernorm (36×) | 0.0056 (max 0.094) |
| post_attention_layernorm (36×) | 0.0005 |
| q_norm / k_norm (72×) | 0.0001 |
| model.norm | 0.0000 (bit-identical) |

### Vocab

151,936 → 151,669 (267 trailing tokens pruned; pruned rows have 2.4× smaller norms — same as 1-bit Bonsai). The 8B README's "151,936" is wrong vs the actual GGUF/unpacked tensors. Note: prune *before* training so embed/lm_head shapes are final.

---

## The reconstructed recipe

1. Start from base FP16 weights as masters. Freeze: embeddings (PTQ-quantize at export), all 1D norms. Trainable: all other 2D Linears + lm_head (reduced LR or shorter schedule for lm_head, judging by its 85% PTQ residue).
2. Forward: BitNet b1.58 STE — `s = mean(|w_group|)` recomputed from the live master every step (NOT a learned parameter), `q = clip(round(w/s)) * s`, gradient straight-through. Group 128 along input dim (= GGUF Q2_0 layout, so export is bit-lossless).
3. Distill against the FP16 teacher long enough for ~70% relative weight movement and ~30% code churn (order 10⁴–10⁵ steps / tens of billions of tokens, judged by movement — this is a training run, not a finetune).
4. No sparsity regularizer, no flip constraints, no mask logits — the 37–40% zeros, per-layer profile, and dead columns all emerge from the absmean rule + gradient pressure.
5. Export: codes + FP16 absmean scales → Q2_0 g128 (2.125 b/w), lossless by construction.

## What this means for the Qwen3.5-9B run (see final chat message for the full prescription)

- Qwen3.5-9B is a hybrid multimodal (`Qwen3_5ForConditionalGeneration`): 32 text layers = 24× gated-DeltaNet `linear_attn` + 8× gated `full_attention` (3:1, head_dim 256), MLP 4096↔12288, vocab 248,320 untied, +1 MTP layer, +27-block vision tower, `mamba_ssm_dtype: float32`.
- Ternarize: mlp.{gate,up,down}_proj (32L), self_attn.{q,k,v,o}_proj (8L, q carries the fused output gate), linear_attn.{in_proj_qkv, in_proj_z, out_proj} (24L), embed (PTQ-frozen), lm_head.
- Islands (FP16/32, frozen): every norm, conv1d (4-tap depthwise), A_log, dt_bias, linear_attn.in_proj_a/in_proj_b (tiny), MTP norms. Vision tower: keep BF16 for v1 (Prism precedent is text-only ternary).
- Text ternary ≈ 9.06B × 2.125/8 ≈ **2.4 GB** (vs ~18.1 GB BF16 text → 7.5×; "ideal" 1.71 b/w framing → 9.4×).

## Artifacts

- `.analysis/bonsai_ternary/deep_full_8b.json` — full per-tensor metrics (this report's source)
- `.analysis/bonsai_ternary/q2_0_forensics_4b.json` — 4B GGUF byte-level stats
- `scratchpad/bonsai_whitepaper.pdf`, `bonsai_1bit_whitepaper.pdf` — no method sections; benchmark methodology in 1-bit Appendix B
