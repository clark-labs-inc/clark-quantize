# Scale-codebook endpoint error attribution

## 2026-07-31 — Post-hoc paired decomposition

This analysis was run only after all three frozen scale-codebook arms had
completed. It is descriptive error attribution, not a preregistered
promotion gate, and it does not authorize final-heldout access or product
promotion. Every comparison uses the same 3,040 development samples.

Changing only the image-I/O scale codebooks from one bit to four bits, while
holding the other category targets at one bit, recovers:

- +0.863996 dB ground-truth PSNR, 95% interval
  `[0.817092, 0.912974]`;
- +1.508850 dB edge PSNR, 95% interval `[1.459338, 1.560557]`;
- +1.540780 dB teacher PSNR, 95% interval `[1.480920, 1.602932]`.

The edge effect wins on 99.61% of paired samples. The two image-I/O
convolutions therefore dominate the edge damage caused by a uniform one-bit
scale channel, despite contributing only 0.22% of the sensitive arm's total
scale-approximation squared error. Aggregate weight error is not a usable
proxy for functional output sensitivity.

Starting from the four-bit image-I/O arm, changing position-bias and
residual-body scale codebooks from one bit to two bits recovers:

- +0.774444 dB ground-truth PSNR, 95% interval
  `[0.719556, 0.829594]`;
- +0.000685 dB edge PSNR, 95% interval `[-0.033172, 0.027170]`;
- +1.483626 dB teacher PSNR, 95% interval `[1.410883, 1.554471]`.

This second bridge materially improves reconstruction and teacher agreement
without a resolved mean edge effect. The two increments exactly decompose the
uniform-to-sensitive aggregate difference because the arms are nested, but
they do not separate position-bias from residual-body contributions. The
ongoing matched RSTB retained-precision study is the next causal localization
test.

### Receipts

- comparison implementation SHA-256:
  `b764a1fe01ccb0340efe5171b1688fb93634ac2abd5a0003be9ca7037d54c9c5`
- image-I/O comparison SHA-256:
  `104f0ced7fecfc66dda0dd8fa278a8759c513b63a5c51a074bc7e2458fff254b`
- position/residual comparison SHA-256:
  `3dc301abfb86133be62cb7314e30ff89a1bf9a882e9c5e4d3a9197ab593e4aaa`

## 2026-08-01 — Final disposition

This study is closed as error attribution. Its localization of image-I/O and
residual sensitivity informed later precision-island work, but it is not a
release candidate and no process is active. G3 is the current selected
artifact; future codebook work must compare from its exact canonical state.
