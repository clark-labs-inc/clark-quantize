# Experiment registry

Each scientific program in this repository has one semantic home. A program
folder should contain:

- `README.md` for scope, acceptance criteria, and folder boundaries;
- `SCIENTIFIC_JOURNAL.md` for dated observations, decisions, negative results,
  and open gates;
- `ARTIFACT_INDEX.md` for canonical code, data, checkpoints, manifests, and
  receipts;
- optional `methods/`, `studies/`, and `policies/` subfolders for stable
  research material.

Journal entries use a dated, evidence-first structure:

1. **Verified** records facts backed by a linked receipt, checkpoint, log, or
   source file.
2. **Interpretation** records what the evidence suggests without promoting it
   to fact.
3. **Decision** records a frozen research choice or acceptance boundary.
4. **Open gate** records the next test that can change the decision.

For an active program, update its journal when a result changes the scientific
picture and update its current-state page when the live execution boundary
changes. Never rewrite an earlier conclusion silently; append a dated
correction.

Live outputs and sealed artifacts are not renamed merely to make the tree
prettier. Their recorded path and hash are part of the scientific evidence.
When a legacy path is still active, its semantic dossier owns it through an
explicit artifact index until the run is frozen and a provenance-preserving
consolidation can be performed.

New scientific programs must start inside their semantic program folder. New
loose experiment outputs must not be added directly under `experiments/`.
An older implementation or run root may remain outside its semantic home only
when its dossier records the exact path, explains the provenance or live-writer
constraint, and defines the safe consolidation gate.

## Current scientific programs

| Program | Semantic home | Legacy or active implementation paths |
|---|---|---|
| BackdropKit Swin2SR browser super-resolution at 1.58 bits | [`swin2sr_super_resolution_1_58bit/`](swin2sr_super_resolution_1_58bit/) | [`browser_swin2sr/`](browser_swin2sr/) |
| LLM ternary and 1.58-bit research | [`llm_ternary_158bit_dossier/`](llm_ternary_158bit_dossier/) | [`large_scale_ternary/`](large_scale_ternary/), [`qwen_agentworld/`](qwen_agentworld/), shared optimizer modules |
| Sana image-generation ternary research | [`sana_image_generation_158bit_dossier/`](sana_image_generation_158bit_dossier/) | Historical Sana run roots recorded in its artifact index |

The Swin2SR program is the active BackdropKit objective. LLM and Sana work are
retained as scientific references and must not redirect the BackdropKit
acceptance gates.
