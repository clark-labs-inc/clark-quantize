# Folder migration record

This document distinguishes completed research organization from unsafe
cosmetic relocation.

## Consolidated into this dossier

- BackdropKit-to-Swin2SR method map:
  `methods/BACKDROPKIT_METHOD_MAP.md`
- sensitive hard-shadow preregistration:
  `studies/SENSITIVE_HARDSHADOW_STUDY_20260730.md`
- promotion-policy amendments:
  `policies/PROMOTION_POLICY_V5_AMENDMENT_20260730.md`,
  `policies/PROMOTION_POLICY_V6_AMENDMENT_20260730.md`, and
  `policies/PROMOTION_POLICY_V7_AMENDMENT_20260730.md`,
  `policies/PROMOTION_POLICY_V8_AMENDMENT_20260730.md`,
  `policies/PROMOTION_POLICY_V9_AMENDMENT_20260731.md`, and
  `policies/PROMOTION_POLICY_V10_AMENDMENT_20260801.md`
- chronological findings and decisions: `SCIENTIFIC_JOURNAL.md`
- canonical evidence map: `ARTIFACT_INDEX.md`
- canonical structured Scientist outputs and run receipts: `scientist_runs/`
- bounded research-agent reports with source identities:
  `research_agent_reviews/`

## Intentionally pinned to `../browser_swin2sr/`

- durably paused sensitive hard-shadow control at step 20,000;
- sensitive-I/O rescue run roots `io-g2-lr025`, `io-g2-lr1`, `io-g2-lr4`,
  and `restart-g9-lr025`; all four are complete through their sealed step-19
  receipts and remain pinned for exact lineage;
- completed optimizer replication receipts whose recorded paths are part of
  their development decision;
- sealed manifests, source snapshots, checkpoints, evaluation receipts, and
  browser-runtime receipts;
- trainer, evaluator, exporter, and runtime files referenced by active process
  commands or sealed source hashes;
- historical negative-result folders whose paths are part of their receipts.

## 2026-08-01 organization audit

All current research is contained by a named program or dated study folder.
There are no live training writers. `../browser_swin2sr/` remains the canonical
implementation and immutable-receipt root because its paths are embedded in
sealed manifests, source closures, published package builders, and runtime
receipts. Moving those bytes would reduce reproducibility rather than improve
organization.

New experiment outputs must start in a dated, semantically named subfolder;
shared trainers, evaluators, and runtimes remain at the implementation root.
Raw datasets, generated media, model payloads, checkpoints, and build caches
are excluded from Git and represented by committed manifests, hashes, ledgers,
and public artifact revisions.

Raw provider responses, hidden reasoning trajectories, credentials, and Codex
session databases are also excluded. Durable structured proposals and reviews
are copied into this dossier; excluded private sources are identified by
run/session ID and content hash where appropriate.

## Intended post-writer layout

After the safe consolidation gate passes, the semantic folder may gain:

- `implementation/` for unsealed Swin2SR-only trainers, evaluators, exporters,
  tests, and browser runtime code;
- `runs/` for future mutable run folders created after migration;
- `receipts/` for future immutable evaluation and runtime receipts;
- `archive/` for classified historical negative-result folders.

Existing sealed folders remain at their original paths and are linked from
`ARTIFACT_INDEX.md`. Consolidation is not permission to rewrite their manifests
or make old commands point at different bytes.

## Safe consolidation gate

No implementation or artifact path is moved until:

1. all writers using `../browser_swin2sr/` have stopped;
2. current command lines, manifests, and process working directories have been
   captured;
3. sealed artifacts have been classified as permanently pinned;
4. every unsealed move has an old path, new path, and SHA-256 receipt;
5. imports, tests, and documentation have been updated together.

This prevents a directory cleanup from changing the identity or
reproducibility of an experiment.
