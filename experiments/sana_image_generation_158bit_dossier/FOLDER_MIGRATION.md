# Folder migration record

This record separates semantic ownership from physical relocation for the
archived Sana image-generation ternary research program.

## Consolidated into this dossier

- research boundary and folder contract: `README.md`;
- verified findings, negative results, unresolved questions, and future gates:
  `SCIENTIFIC_JOURNAL.md`;
- canonical evidence, implementation, data, and release map:
  `ARTIFACT_INDEX.md`.

These files are the program's navigation and interpretation layer. New
scientific synthesis for this inactive lane belongs here.

## Intentionally pinned to historical paths

- root-level Sana reports and cross-model research syntheses;
- quantizers and shared QAT implementations under `../../clark_quantize/`;
- historical Sana trainers, packers, renderers, and publishing helpers under
  `../../scripts/`;
- evaluation and ranking records under `../../data/`;
- model cards and loaders under `../../release/`;
- release manifests under `../../releases/`;
- checkpoints, logs, and artifact repositories identified by recorded remote
  paths.

Moving those files would change historical commands, Python import roots,
release references, or artifact provenance. Their authoritative locations are
listed in `ARTIFACT_INDEX.md`.

## Safe relocation gate

A historical file may move into this dossier only when:

1. no active or reproducibility command depends on its current path;
2. repository-wide imports, release references, and links are enumerated;
3. the old and new paths plus the pre-move SHA-256 are recorded;
4. all callers and documentation are updated in the same change;
5. published manifests and external artifact records do not identify the old
   path.

Until every condition holds, indexing is the provenance-preserving form of
consolidation. This archived lane must not be resumed without a fresh explicit
decision and a new dated study.
