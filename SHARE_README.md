# Bonsai-Clark: Prism-style QAT + Clark PTQ

Reverse-engineered from Prism ML's Bonsai checkpoints and validated on a tiny DiT.

## What this is

1. **`prism_weight_analysis.md`** — Full forensic report from analyzing Prism's actual model weights (ternary/binary image + LLM checkpoints, diffed against base models). Contains the 10 key findings that reveal their training method.

2. **`bonsai_quant.py`** — QAT library implementing the reverse-engineered method:
   - `TernaryLinear`: STE ternary wrapper (master weight + hard ternary forward)
   - `SignLockedTernaryLinear`: sign-locked variant (frozen signs, trainable mask + scale)
   - `ProximalPullScheduler`: drag master weights toward codebook
   - `wrap_model_ternary`: replace nn.Linear with TernaryLinear, protect islands
   - Forensic utilities: code stats, ternary error, param budget

3. **`tiny_dit.py`** — Minimal DiT (1.8M params) for 32×32 images with flow matching.

4. **`train_bonsai.py`** — Full pipeline: train FP16 teacher → PTQ baseline → Prism-style QAT → compare → analyze. Includes synthetic data fallback.

5. **`clark_quantize/`** — Original Clark PTQ library (data-free optimal-scale ternary + GPTQ + incoherence rotation). This is the initialization method that Prism also uses.

6. **`scripts/`** — Sana-specific training scripts (PrismLock QAT, CTD, mirror training) + utilities.

## Key findings from Prism weight forensics

### Image model (FLUX.2 Klein 4B → Bonsai Image 4B)
- All 100 trunk tensors: exactly ternary (0.0000% error at g128)
- Trunk moved 54% from base (3-5× more than PTQ → confirms training)
- 0.002% sign flips (STE preserves signs)
- `proj_out` bit-identical to base (FROZEN — random-features last layer)
- `norm_out` moved 141% (island compensation channel)
- Block modulation barely moved (1.5%)
- Scales shrank to 63% of PTQ, zero fraction 48%→38% (denser)
- 5% island at FP16, 95% trunk at ternary

### LLM model (Qwen3-8B → Ternary-Bonsai-8B)
- ALL 396 tensors ternary including embeddings and LM head (0.0000% error)
- NO islands except 308K params of 1D RMSNorm (0.004%)
- Trunk moved 75% from base (more aggressive than image)
- 1.09% sign flips (NOT zero — LLM explores codebook more)
- `model.norm` frozen at 0% (no compensation island)
- Late-layer scales GROW (1.21×) — self-compensation without island
- 9.35× compression (ternary) vs image's 6.59× (no island overhead)

### Universal across both
- STE QAT + self-distillation + optimal-scale PTQ init + ternary g128
- Delta increases with depth
- Zero fraction decreases (training activates weights PTQ zeros out)
- Activation balanced (equal 0→+1 and 0→-1)
- Ternary and binary separately trained
- 1D norms barely move

## Validated on TinyDiT

| Model | Velocity MSE | Trajectory MSE |
|---|---|---|
| Teacher (FP16) | 0.486 | 0.190 |
| PTQ ternary | 0.632 | 0.225 |
| QAT ternary | 0.494 | 0.190 |

QAT recovered teacher performance (1.6% gap on velocity, 0% on trajectory).
Same code patterns: 0% sign flips, 98.7% code preservation, proj_out frozen.

## Usage

```bash
pip install -r requirements.txt

# Full pipeline on TinyDiT (synthetic data, ~2 min on MPS)
python train_bonsai.py --phase all --dim 128 --depth 4 --heads 4 \
  --teacher-epochs 5 --qat-steps 500 --batch-size 64 --device mps

# Or phase by phase
python train_bonsai.py --phase teacher  # train FP16 teacher
python train_bonsai.py --phase qat      # Prism-style QAT
python train_bonsai.py --phase analyze  # forensic analysis
python train_bonsai.py --phase sample   # generate images
```

## The method (one paragraph)

Initialize from optimal-scale ternary PTQ → wrap trunk as TernaryLinear (master weight `u`, hard ternary forward via STE) → freeze `proj_out` → train against FP16 teacher with velocity + trajectory distillation loss → proximal pull `u ← (1-ρ)u + ρŴ` after each optimizer step → STE preserves signs automatically → training discovers which zeros to activate and how to shrink scales → bake codes → pack.
