# Sana Image-Generation 1.58-bit Research Dossier

Status: archived scientific reference
Last reviewed: 2026-07-30

This folder is the self-contained navigation and interpretation surface for
Clark's Sana 1.6B low-bit image-generation experiments. Sana is not an active
training lane. BackdropKit's tiny in-browser models, beginning with Swin2SR,
are the current optimization target.

## Folder contents

- [Scientific journal](SCIENTIFIC_JOURNAL.md) — verified findings, stable
  negative results, unresolved questions, and the gates required before this
  lane could be resumed.
- [Artifact index](ARTIFACT_INDEX.md) — exact links to authoritative release
  metadata, reports, quantizers, trainers, evaluation data, and runtime
  surfaces.
- [Folder migration record](FOLDER_MIGRATION.md) — files consolidated here,
  historical paths intentionally preserved, and the safe relocation gate.

## Provenance policy

Historical reports, source files, datasets, release metadata, checkpoints, and
logs remain at their original repository or recorded remote paths. They were
not copied or moved into this folder because doing so would break old commands,
imports, release paths, cross-references, or artifact provenance. This dossier
is therefore self-contained as an evidence map, while the linked run-specific
receipt is authoritative for every numerical claim.

Do not infer current process, checkpoint, or remote-storage state from this
archive. Reopen the live source before reporting current status.

## Research boundary

The transferable mechanisms retained for BackdropKit research are
MSE-sensitive initialization, continuous master weights, hard ternary
forwards, protected sensitive surfaces, gradual low-bit schedules, bounded
optimizer state, and separate quality and packing gates. No Sana checkpoint or
Sana-specific training objective is part of the active Swin2SR program.
