# Scientist run registry

This directory is the canonical Git home for durable Clark Scientist artifacts
that concern the BackdropKit Swin2SR program.

Each run folder retains the complete research surface when repository
visibility and source ownership permit it:

- the exact structured proposal or other model-visible output;
- raw provider responses, reasoning details, worker configuration, event
  streams, and trajectories;
- a receipt naming the model, authority, source run, hashes, credential scan,
  and any exclusions;
- links to the deterministic experiment evidence that accepted or rejected the
  proposal.

Credentials and unrelated machine-private data are never committed. Codex
session databases and Codex internal reasoning are not Clark Scientist model
artifacts and remain outside this registry. Their terminal research reports
may be retained separately under `research_agent_reviews/`.

The Clark Scientist engine remains a separate software repository. This
directory owns its BackdropKit research outputs.
