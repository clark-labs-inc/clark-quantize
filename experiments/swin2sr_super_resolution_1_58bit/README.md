# Swin2SR browser super-resolution at 1.58 bits

This folder is the scientific dossier for reducing BackdropKit's tiny
Swin2SR x2 super-resolution model to a mainly ternary, Bonsai-style artifact
with explicitly protected precision islands and no hidden storage cost. The
historical 1.58-bit target remains useful for the ternary trunk, but whole-model
promotion is governed by measured quality and honest lossless bits per model
parameter rather than forcing every layer into exactly 1.58 bits.

The canonical implementation and immutable experiment outputs currently live
in `../browser_swin2sr/`. That directory contains active writers and sealed
manifests whose paths and source hashes are evidence. They must not be renamed
or moved while runs are active, and sealed artifacts should remain at their
recorded paths afterward. This dossier owns the research semantically through
an append-only journal and an explicit artifact index.

## Folder contract

This directory is the single scientific home for the Swin2SR program:

- discoveries and decisions are recorded in this folder;
- stable study, method, and policy documents live in this folder;
- code and immutable receipts are addressed from `ARTIFACT_INDEX.md`;
- the legacy `../browser_swin2sr/` path is an active implementation and
  artifact root, not a second research program.

No writer is active as of 2026-08-01. The implementation root remains pinned
because sealed commands, manifests, hashes, imports, published packages, and
external receipts record those paths. Consolidation may move only unsealed
implementation files after a path-and-hash migration receipt is prepared.
Sealed run folders remain at their recorded paths.

## Contents

- `SCIENTIFIC_JOURNAL.md` — chronological decisions, results, failures, and
  open gates.
- `CURRENT_STATE.md` — concise live-program boundary and next scientific
  gates; reopen the linked processes and receipts before quoting it later.
- `ACCEPTANCE_LEDGER.md` — requirement-by-requirement proof ledger for the
  actual BackdropKit model; infrastructure demonstrations cannot close
  candidate-quality gates.
- `ARTIFACT_INDEX.md` — authoritative paths for datasets, baselines, run
  manifests, receipts, code, and runtime evidence.
- `methods/` — stable method maps and cross-experiment lessons.
- `studies/` — preregistered study descriptions.
- `policies/` — promotion-policy amendments.
- `FOLDER_MIGRATION.md` — what has already been consolidated and what remains
  intentionally pinned to a legacy path.

## Scientific rules

1. Start from the non-quantized Swin2SR teacher, never from the shipped q4
   student.
2. Train continuous master weights progressively toward hard ternary
   endpoints, with sensitive surfaces scheduled separately.
3. Evaluate the hard endpoint during training; blended quality is not
   promotion evidence.
4. Development bundles may choose recipes. The sealed glyph-safe heldout may
   be opened only for a candidate that passes all preregistered development
   gates.
5. A candidate claiming q4 replacement must be statistically non-inferior to q4, materially smaller,
   and runnable through WebGPU, WASM, and CPU fallbacks before integration.
6. Infrastructure-only artifacts and browser receipts never count as visual
   quality evidence.
7. Every journal entry distinguishes verified evidence from interpretation.
