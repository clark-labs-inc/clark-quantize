# Folder migration record

This record separates semantic ownership from physical relocation for the
archived LLM ternary and 1.58-bit research program.

## Consolidated into this dossier

- research boundary and folder contract: `README.md`;
- verified findings, negative results, unresolved questions, and future gates:
  `SCIENTIFIC_JOURNAL.md`;
- canonical evidence and implementation map: `ARTIFACT_INDEX.md`.

These files are the program's navigation and interpretation layer. New
scientific synthesis for this inactive lane belongs here.

## Intentionally pinned to historical paths

- `../qwen_agentworld/` run records, trainers, cache builders, protocols, and
  packed-runtime experiments;
- `../large_scale_ternary/` frozen Gate B protocol, evaluation universe, and
  optimizer audit;
- root-level Qwen run reports, recipes, runbooks, and cross-model compression
  analyses;
- shared optimizer and quantization modules under `../../clark_quantize/`;
- checkpoints and logs referenced by recorded local or remote paths.

Moving those files would change historical commands, Python import roots,
cross-references, or checkpoint provenance. Their authoritative locations are
listed in `ARTIFACT_INDEX.md`.

## Safe relocation gate

A historical file may move into this dossier only when:

1. no active or resumable command depends on its current path;
2. repository-wide imports and links have been enumerated;
3. the old and new paths plus the pre-move SHA-256 are recorded;
4. all callers and documentation are updated in the same change;
5. frozen manifests and external artifact records do not identify the old path.

Until every condition holds, indexing is the provenance-preserving form of
consolidation. This archived lane must not be resumed without a fresh explicit
decision, model/provider confirmation where applicable, and a new dated study.
