# Clark Air — deployable kernel formats research notes

Goal: produce Prism-style deployable artifacts for **Clark Air Sana 1.6B** — `gemlite-2bit`,
`gemlite-1bit`, `mlx-2bit`, `mlx-1bit` — i.e. kernel-runnable quantized weights, not just
disk-compressed files. Source model: our QAT ternary Sana (master-weight QAT, g112).

## TL;DR status

| Artifact | Status | Notes |
|---|---|---|
| **gemlite-2bit** | ✅ built + render-verified (CUDA) | ternary re-quantized from master weights at per-layer pow2 groups |
| **mlx-2bit** | ✅ transformer built + verified (Apple Silicon) | first pure-MLX Sana; our ternary injected into MLX 2-bit kernels |
| **gemlite-1bit** | ⛔ needs binary model | separate binary QAT run (blocked on GPU) |
| **mlx-1bit** | ⛔ needs binary model | same |
| full standalone MLX pipeline | ⏳ | DPM scheduler + Gemma-2 text enc + DC-AE VAE still to port |

## Hard constraints discovered

1. **2-bit = ternary, 1-bit = binary.** Our model is ternary {-1,0,+1} (≈1.58-bit). The
   1-bit (`*-1bit`) artifacts are *binary* and require a **separate binary training run**
   (Prism's data: binary ~88% quality vs ternary ~95%). Cannot be faked from the ternary model.

2. **Kernels need power-of-two groups; Sana's dims fight it.** Our model is quantized at
   **g112** (because 2240 = 20×112). gemlite/MLX 2-bit kernels require power-of-two groups.
   2240 = 2⁶·5·7 → max pow2 group **64**; the GLU FFN's 5600 = 2⁵·5²·7 → max pow2 **32**.
   So every layer is re-quantized at its **own** largest pow2 group (g64 / g32).
   - Re-quantizing the *released g112 output* down to g128/g64 loses ~7% per layer (double-quant).
   - Re-quantizing the *continuous master weights* at g64/g32 is the right path (normal ~4%
     ternary error). Master weights live in the QAT state file as `<module>.weight_master`.

3. **MLX has no Sana runtime.** mflux/DiffusionKit cover FLUX/SD, not Sana. So MLX work was a
   from-scratch port of Sana's transformer (and eventually scheduler/text-enc/VAE).

4. **MLX native `mx.quantize` 2-bit is naive affine → garbage on the trunk** (127% error; its
   4-level grid doesn't even include 0). Our *trained* ternary must be injected into MLX's
   2-bit slots manually.

## gemlite-2bit (`scripts/pack_sana_gemlite.py`)

- gemlite path: `GemLiteLinear(W_nbits=2, group_size=g)` + `.pack(Wq, scales, zeros)`.
  Ternary as 2-bit: `Wq = code+1 ∈ {0,1,2}` (4th level unused), `zeros = 1` (→ dequant
  `scale*(Wq-1) = scale*code`), `fma_mode=False`. Verified exact (3e-4) at g64/g32.
- `A16W158_INT` helper is **per-tensor** scale (BitNet style) — does NOT fit our per-group
  scales. Use the general W2 path instead.
- Build: load `weight_master` from the QAT state (step ~43679), re-quant MSE-optimal ternary
  at per-layer pow2 group, pack gemlite, replace Sana's Linear (attn) + 1×1 Conv (FFN) with
  gemlite layers (`GemliteLinearWrap` / `GemliteConv1x1Wrap`), keep islands fp16.
- Result: 200 trunk modules packed (180×g64, 20×g32), renders near-FP16 on real GemLite INT2
  kernels. Larger on disk than the 374 MB packed file (g64/g32 + fp16 scales/islands ≈ ~480 MB)
  but it is the *runtime* format.

## MLX port (`scripts/mlx_sana.py`)

First pure-MLX Sana 1.6B transformer. Architecture ported by hand from diffusers
`SanaTransformer2DModel`:
- **ReLU linear self-attention**: q,k → relu; value augmented with a ones row;
  `scores = v·k`, `out = scores·q`, normalize by the last (ones) channel.
- **SDPA cross-attention** (caption), with `encoder_attention_mask` (caption is padded to 300;
  valid tokens ~1 uncond / ~14 cond — masking is essential).
- **GLU-MBConv FFN**: 1×1 conv_inverted (quantized) → SiLU → 3×3 depthwise conv_depth (island,
  MLX NHWC) → gated split → 1×1 conv_point (quantized).
- AdaLN-single modulation (`scale_shift_table` + timestep), caption projection (gelu-tanh),
  RMSNorm caption_norm, sinusoidal timestep embedding.
- **MLX 2-bit packing**: little-endian, 16 two-bit values per uint32; scales/biases `[out, in/g]`.
  Inject ternary as `levels = code+1`, `scale = s`, `bias = -s` → affine dequant reproduces
  `code*s` exactly (unit test 3e-4). `mx.quantized_matmul(..., bits=2, group_size=g)`.

### Verification (vs PyTorch reference, loading our trained weights)
- Single random input: fp16 port **1.2%** rel error.
- **Over the real 20-step generation trajectory** (CFG, real prompt, correct masking):
  - fp16 port: mean **0.15%**, max 0.18% (near-exact).
  - mlx-2bit (ternary injected): mean **3.4%**, max 4.2% (ternary level → renders like FP16).
- Forward ~1.4 s on M5 Max.

## Open items / next steps
- Build the *faithful* mlx-2bit / gemlite-2bit from master weights (drops the ~3% double-quant
  when starting from the unpacked release).
- Full standalone MLX pipeline: port DPM-Solver multistep scheduler, Gemma-2 text encoder,
  DC-AE VAE → render end-to-end in MLX (the transformer was the hard part).
- Binary QAT run at a gemlite/MLX-native group (g32 divides all Sana dims) → `gemlite-1bit` /
  `mlx-1bit`. Blocked: the a100 GPU is currently held by another project.
