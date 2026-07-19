# Clark Air Universal Compressor

## Current Conclusion

The Sana work established the core mechanism, but the image-specific objective
branch is not the mainline anymore.

The current PTD/LATENTS=4 continuation was stopped and archived as a stable
plateau probe:

- tag: `sana_t112_ptd1_lat4_s43576_to100k`
- stopped: `s48438`
- archive: `/workspace/archive_sana_t112_ptd1_lat4_s43576_to100k_s48438_plateau`
- verdict: stable semantic-tangent probe, no visual breakthrough
- scalar band: `L ~= 0.244-0.249`
- quant stats: `zero ~= 0.477`, `flip = 0.0000`, `scale ~= 0.219`

The next mainline is:

```text
Clark Air:
universal binary/ternary transformer compression
via teacher-behavior QAT + rate-distortion allocation
```

Sana remains a useful DiT testbed, but it should not be the only target or the
primary optimization objective.

## Stopped Runs To Treat As Stable Negatives

The later Sana objective probes should be kept as evidence, not continued as
the mainline:

- Replay / x0 / perceptual: preserved broad structure but pushed toward soft,
  averaged detail.
- Bellman / value shaping: technically viable, but mostly learned trajectory
  closeness rather than visible object/relation correctness.
- PTD: reached the strongest scalar band around `L ~= 0.244`, but did not
  produce a clear render champion.
- Stein / Jeffreys teacher-behavior run:
  `sana_t112_air_teacher_behavior_s43679_v1` stopped at `s46883`; stable,
  no collapse, no visual breakthrough.
- No-checkpoint A100 speed probe: failed at first train step because the full
  teacher-copy recipe consumed nearly the entire 80GB device. Gradient
  checkpointing is required for this image setup unless the train loop is
  rebatched.

These runs support the pivot: Sana proved hard-ternary QAT works, but further
Sana-only objective exploration is lower return than building the universal
compressor.

## Evidence From Sana

1. Master-weight QAT is the correct core.
   Hard binary/ternary weights must be used in the forward pass, while
   continuous master weights receive STE gradients.

2. PTQ is only an initializer.
   MSE-optimal ternary quantization is the correct data-free baseline, but it
   does not by itself produce full-quality low-bit models.

3. Explicit support editing was the wrong abstraction.
   CTD, Mirror-style support freezing, explicit mask variables, density losses,
   and forced support movement repeatedly damaged quality or failed to move the
   useful structure.

4. QAT moves the model into the codebook.
   The successful branches keep support derived from moving master weights
   rather than directly editing code variables.

5. Pure binary is not the image product route.
   Prior binary/ternary grids showed that ternary preserved hard compositions
   while binary collapsed on the harp and horse/person prompts. The 8x route is
   therefore mixed-rate, not global binary.

6. Image-only objectives plateau.
   Replay, perceptual, Bellman, self-oracle, Stein/Jeffreys, CVF, and PTD probes
   were informative but did not reliably beat the strongest science-fresh
   visual branch.

## Universal Objective

For a teacher transformer `T` and hard-low-bit student `Q`, optimize:

```text
min_W  D_behavior(W) + lambda * B(W)
subject to W in binary/ternary/selected-island codebooks
```

Where:

- `D_behavior` is teacher-behavior distortion.
- `B(W)` is the actual packed bit cost.

For LLMs:

```text
D_behavior =
  KL[p_T(. | teacher prefixes) || p_Q(. | teacher prefixes)]
+ KL[p_T(. | student prefixes) || p_Q(. | student prefixes)]
+ hidden-state geometry loss
```

For diffusion transformers:

```text
D_behavior =
  E_{x_t ~ teacher path} ||v_Q(x_t,t,c) - v_T(x_t,t,c)||^2
+ E_{x_t ~ student path} ||v_Q(x_t,t,c) - v_T(x_t,t,c)||^2
+ hidden-state geometry loss
```

For VLMs:

```text
D_behavior =
  text-token KL
+ image-token feature loss
+ multimodal hidden-state geometry
+ task-specific output KL when available
```

All variants share codebook geometry:

```text
L_code = mean_i ((u_i - Q(u_i))^2 / (s_group(i)^2 + eps))
```

## Rate-Distortion Allocation

For each weight group `g` and candidate rate:

```text
r in {binary, ternary, fp8/island}
```

Estimate behavioral damage using a local Fisher/Hessian approximation:

```text
DeltaD_{g,r} ~= 0.5 * DeltaW_{g,r}^T F_g DeltaW_{g,r}
```

Then choose the rate by water-filling:

```text
r_g(lambda) = argmin_r [DeltaD_{g,r} + lambda * bits_{g,r}]
```

Increase `lambda` until:

```text
sum_g bits_{g,r_g} <= target_bits
```

For MoE models, scale each expert group's Fisher by router utilization:

```text
F_effective(e,g) = p_expert(e) * F(e,g)
```

This spends bits on hot/sensitive experts and compresses cold experts more
aggressively.

## Near-Term Implementation Plan

1. Freeze the Sana image branch.
   Keep the science-fresh checkpoint family as the current visual reference.
   Archive PTD, Stein, Bellman, replay, QAT8, and mixed-rate image probes as
   research evidence.

2. Build a general Clark Air core.
   Reusable components should live under `clark_air/`:

   ```text
   clark_air/
     quant/
       optimal_ternary.py
       binary.py
       master_weight_qat.py
       packing.py
     objectives/
       diffusion_teacher.py
       llm_teacher_kl.py
       hidden_geometry.py
       codebook_geometry.py
     rate/
       ledger.py
       waterfill.py
       fisher_diag.py
       mixed_rate_policy.py
     train/
       train_diffusion_qat.py
       train_llm_qat.py
     policies/
       sana.yaml
       flux.yaml
       dense_llm.yaml
       moe_llm.yaml
   ```

3. First universal proof: small dense LLM.
   Start with a 1B-3B dense model before GLM-scale MoE. Validate:

   - ternary QAT beats ternary PTQ
   - mixed binary/ternary reaches 8x-class storage
   - perplexity/KL loss is small
   - packed bytes match the ledger

4. Second proof: MoE allocation.
   Use router utilization times Fisher damage to choose binary/ternary rates per
   expert group.

5. Third proof: DiT transfer.
   Re-run the method on Sana or a FLUX-style DiT only after the general
   compressor path is established.

## Working Rule

Do not choose binary or ternary globally.

Choose it per group/module/expert by teacher-behavior sensitivity and bit cost.
