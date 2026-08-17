# Direct radix-5 AVX2 decision record

The first direct-compute series is closed without promotion. All five variants
used the same real Qwen3.8-27B `gate_proj` matrix, the same aligned-2 control,
the same 1.75-bit radix-5 weights including FP16 group scales, the same cache
eviction and core placement, and frozen correctness/performance gates.

| Variant | Design | 1 core | 4 cores | Gate |
|---|---|---:|---:|---|
| v1 | AVX2 gather LUT | 18.55% | 36.99% | fail |
| v2 | ordinary-load LUT | 38.23% | **68.22%** | fail |
| v3 | static ordinary-load LUT | 36.87% | 67.07% | fail |
| v4 | arithmetic digit planes | 27.63% | 51.73% | fail |
| v5 | paired-row ordinary-load LUT | 28.14% | 52.63% | fail |

Percentages are median elapsed-time throughput relative to the paired
aligned-2 kernel, not ratios of their differently sized byte streams. All
variants passed exact pack/unpack, numerical, and thread-determinism checks.
None reached the preregistered 70% minimum, and none was a Pareto speed win.

## Decision

- Keep canonical base-3/radix-5 as the dense storage and transport form.
- Keep the 2.125-bit aligned-2 group-128 image as the promoted AVX2 compute
  form. Transcoding at load time is allowed and must be included in startup
  accounting.
- Do not integrate direct radix-5 into the model graph on the strength of v2's
  near miss.
- A future dense compute branch must change the machine contract: e.g. an
  AVX-512 VBMI lookup design, an architecture-specific dot-product path, or a
  block code jointly designed for native shuffles. It must preregister against
  v2 and the aligned-2 control.
- Kernel-only exact arithmetic experiments do not require language capability
  evaluation. Any graph-integrated/model-changing candidate must run the
  mandatory WikiText, MATH-500, and CUTE/MMLU quick gate before longer work.

No scientific-breakthrough claim is authorized by this series.
