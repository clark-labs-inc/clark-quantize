# Numerical-conformance failure

V3 emitted a 922,951-byte affine artifact but failed packed numerical
conformance (`51.12` dB versus the exact evaluated state). Root-cause replay
found that CPU projection flips one near-threshold ternary code in each of five
core tensors. Reconstructing the same frozen checkpoint on MPS reproduces all
484 evaluated tensors exactly with zero differing values.

The v3 artifact is therefore a CPU-reprojected variant, not the measured
35.260 dB candidate, and must not be reused or promoted. V4 requires the
materializer's source device to reproduce the evaluated tensor SHA before it
is allowed to write the affine payload.
