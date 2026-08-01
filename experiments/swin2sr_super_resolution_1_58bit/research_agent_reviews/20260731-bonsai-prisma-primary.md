# Bonsai and Prism precision topology

- Source task: `/root/bonsai_prisma_primary`
- Source session: `019fb948-ba8e-7c60-9662-84136d90ede7`
- Captured: 2026-08-01
- Status: archival research lead; verify claims against primary evidence before use


Canonical evidence: [Prism/Bonsai artifact index](../../browser_swin2sr/prism_bonsai_transfer_20260731/ARTIFACT_INDEX.md) and [packed-weight inventory](../../browser_swin2sr/prism_bonsai_transfer_20260731/packed-weight-inventory.json).

## Report

Research complete; I sent the parent exact PrismML model identifiers, pinned checkpoint revisions, precision topology, byte-level accounting, and Swin2SR hypotheses.

Key discovery: Bonsai Image does not quantize everything. It ternarizes 94.97% of transformer parameters while retaining 5.03% in BF16. Those protected tensors are mostly trainable modulation, embedding, normalization, and output-path layers—not merely frozen copies of the base model.

I also identified a runtime opportunity: the MLX ternary artifact stores both `scale` and an affine `bias` that equals `-scale`. A custom BackdropKit format can reconstruct that bias instead of storing it, then optionally use tighter trit packing and quantized scale codebooks.

The public Bonsai Image materials do not disclose their training data, optimizer, loss, or gradual ternarization schedule, so any claim about their exact training recipe would be speculation.
