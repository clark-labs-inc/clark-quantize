# LLM 1.58-bit Ternary Research Dossier

Status: archived scientific reference
Last reviewed: 2026-07-30

This folder is the self-contained navigation and interpretation surface for
Clark's LLM ternary-quantization experiments. The large-model program is not an
active training lane. BackdropKit's tiny in-browser models, beginning with
Swin2SR, are the current optimization target.

## Folder contents

- [Scientific journal](SCIENTIFIC_JOURNAL.md) — verified findings, negative
  results, unresolved questions, and the gates required before this lane could
  be resumed.
- [Artifact index](ARTIFACT_INDEX.md) — exact links to authoritative run
  reports, frozen evaluation protocols, implementation surfaces, and
  scientific synthesis.
- [Folder migration record](FOLDER_MIGRATION.md) — files consolidated here,
  historical paths intentionally preserved, and the safe relocation gate.

## Provenance policy

Historical reports, source files, protocols, checkpoints, and logs remain at
their original repository or recorded remote paths. They were not copied or
moved into this folder because doing so would break old commands, imports,
cross-references, or artifact provenance. This dossier is therefore
self-contained as an evidence map, while the linked run-specific receipt is
authoritative for every numerical claim.

Do not infer current process, checkpoint, or remote-storage state from this
archive. Reopen the live source before reporting current status.

## Research boundary

The transferable mechanisms retained for BackdropKit research are continuous
master weights, hard low-bit forwards with straight-through gradients,
sensitive-surface scheduling, bounded optimizer state, behavior-level gates,
and separation of packing efficiency from model quality. No LLM checkpoint or
LLM-specific objective is part of the active Swin2SR program.
