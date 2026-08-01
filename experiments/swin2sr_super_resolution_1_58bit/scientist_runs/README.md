# Scientist run registry

This directory is the canonical Git home for durable Clark Scientist artifacts
that concern the BackdropKit Swin2SR program.

Each run folder retains the reproducible scientific surface:

- the exact structured proposal or other model-visible output;
- a receipt naming the model, authority, source run, hashes, and exclusions;
- links to the deterministic experiment evidence that accepted or rejected the
  proposal.

Private provider trajectories, hidden reasoning, credentials, caches, and
machine-specific session databases are not Git artifacts. They remain outside
the repository. A receipt records the identity of any excluded raw run needed
for a local audit.

The Clark Scientist engine remains a separate software repository. This
directory owns its BackdropKit research outputs.
