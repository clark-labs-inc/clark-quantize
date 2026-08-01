# Data-scale and degradation study

Status: proposed; not sealed or started
Selection source: development only
Final heldout: locked

## Question

Does increasing the clean-image corpus and/or broadening the low-quality
degradation distribution improve the fully ternary Swin2SR candidate after
the current sensitive-I/O and optimizer decisions are complete?

## Current evidence

- The current balanced epoch contains 206,500 distinct image files:
  104,500 UI/general images and 102,000 glyph-safe synthetic UI/text images.
- The current 30,000-step schedule at batch size 8 consumes 240,000 training
  draws, only about 1.16 balanced epochs.
- The degradation function is currently bicubic x2 plus JPEG quality 55-92
  with probability 0.35. It does not model browser scaling, alternative
  resampling kernels, WebP/AVIF, video codecs, chroma subsampling, blur,
  ringing, sharpening, noise, or display capture.
- The current output-convolution study shows that fully hard representation
  capacity is a bottleneck. More data may improve recovery and generalization,
  but cannot be assumed to remove a structural quantization bottleneck.
- The published Ternary Bonsai materials disclose the ternary format and
  benchmarks, but not a training dataset size or sufficient training recipe
  to attribute its quality to a specific corpus scale.

## Hypotheses

H1. A richer, product-demo-realistic degradation distribution improves
development edge, teacher, and worst-decile metrics even at the current
206,500-image corpus size.

H2. Increasing clean source diversity improves generalization after controlling
for degradation and optimizer updates.

H3. Data and degradation interact: a million-source corpus with a narrow
bicubic/JPEG degradation distribution will underperform the same corpus with a
realistic browser/video degradation distribution.

H4. Portrait-heavy expansion is inefficient for BackdropKit. People and skin,
hair, and fabric should be a bounded robustness slice, while screenshots,
product surfaces, typography, and product-demo video frames remain dominant.

## Candidate corpus

Build immutable, licensed manifests rather than an untracked scrape. The first
one-million-source target should be weighted toward the product:

- 55-65% UI screenshots, product screenshots, synthetic sites, and application
  surfaces;
- 15-20% multilingual typography, charts, diagrams, icons, and high-contrast
  synthetic UI, balanced across the existing 204-language contract;
- 10-15% public-domain/CC0 product, object, texture, and general photography;
- 5-10% product-demo video frames and screen recordings, deduplicated by scene;
- at most 5% portraits/people as a robustness slice, not the training center.

Prefer expanding already-receipted sources before introducing new licensing
surfaces:

- WebSight provides up to two million synthetic screenshot/HTML pairs;
- PD12M provides 12.4 million public-domain/CC0 image-caption pairs;
- Aria-UI provides additional desktop UI data, but its full set is noisier and
  must pass overlap, window-occlusion, and quality filters;
- LSDIR provides 84,991 high-resolution restoration images for texture and
  natural-image coverage.

Every included item must retain source revision, original identifier, license,
content hash, dimensions, and all filtering decisions. Perform exact hash,
perceptual hash, and embedding-near-duplicate isolation against development and
the unopened final heldout before training.

## Online degradation distribution

Keep clean high-resolution sources and generate low-resolution pairs
deterministically from a recorded seed. Include:

- bicubic, Lanczos, bilinear, and area downsampling with antialias variations;
- isotropic and anisotropic blur, mild defocus, ringing, and sharpening halos;
- JPEG quality and 4:4:4/4:2:2/4:2:0 chroma paths;
- WebP and AVIF still-image compression where the pinned runtime supports it;
- H.264, HEVC, VP9, and AV1 product-video frame round trips at several bitrates;
- browser/device-pixel-ratio rescaling and fractional CSS-scale capture;
- low-amplitude sensor/display noise, gamma, and color-space perturbations;
- a clean bicubic anchor slice to prevent regression on the shipped use case.

Store degradation parameters in the sampler state so exact resume and paired
arm comparison remain possible. Do not materialize a million low-quality
copies when deterministic online generation can produce many independent
pairs from each clean source.

## Preregistered ablation shape

Do not mutate or reopen the current output-convolution or optimizer studies.
After their winner is frozen, fork the exact same fully hard parent and compare:

| Arm | Clean corpus | Degradation |
|---|---:|---|
| A | 206,500 | current |
| B | 206,500 | expanded |
| C | approximately 1,000,000 | current |
| D | approximately 1,000,000 | expanded |

Use identical initialization, optimizer state, sample count, batch size,
schedule, and validation cadence. The first decisive pass should expose each
arm to one million training draws; at batch size 8 this is 125,000 steps.
Checkpoint at matched draw counts and allow an early futility stop only under a
predeclared paired rule. Repeat the retained arms with independent sampler
seeds before selection.

The selector must use paired development overall, edge, teacher, worst-decile,
source, language, and script metrics, plus a product-demo-specific slice. Data
scale is accepted only if the confidence interval and every protected slice
meet the frozen non-inferiority margin. The final heldout remains unopened
until the complete model, artifact, and runtime candidate is frozen.

## References

- Swin2SR paper: https://arxiv.org/abs/2209.11345
- Binary super-resolution paper:
  https://proceedings.neurips.cc/paper_files/paper/2024/file/3685de48976169ca9fd68cb4c8e48b76-Paper-Conference.pdf
- WebSight: https://arxiv.org/abs/2403.09029
- PD12M: https://arxiv.org/abs/2410.23144
- Aria-UI dataset:
  https://huggingface.co/datasets/Aria-UI/Aria-UI_Data
- LSDIR: https://ofsoundof.github.io/lsdir-data/
- Ternary Bonsai whitepaper:
  https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf
