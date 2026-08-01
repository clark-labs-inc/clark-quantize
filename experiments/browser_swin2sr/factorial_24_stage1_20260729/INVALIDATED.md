# Invalidated factorial run

This directory is preserved as negative evidence and must not be used to rank
optimizer arms.

The stage-one launch applied `scheduled_prox_rho` to every optimizer except
plain AdamW. Consequently, both `bonsai_qat8` and `bonsai_qat8_muon` combined
their intended optimizer with the historical proximal pull, while their names
and preregistration described optimizer-only controls. This confounded the
factorial comparison.

The issue was detected from the emitted validation records, where the Bonsai
arms reported nonzero `prox_rho`. The run was stopped without deleting or
overwriting its checkpoints. A clean replacement uses a new output directory
and makes the proximal schedule exclusive to the `adamw_prox` arm.
