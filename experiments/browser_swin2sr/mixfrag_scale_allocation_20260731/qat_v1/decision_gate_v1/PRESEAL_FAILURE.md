# Pre-seal failure

This decision attempt stopped before writing `decision.json` because the
frozen decision-source set omitted `training_data_contract.py`, a direct local
dependency of `full_surface_provenance.py`.

The trained checkpoint, materialization, development report, and artifact are
unchanged. A new versioned decision gate must include that dependency in its
sealed source set.
