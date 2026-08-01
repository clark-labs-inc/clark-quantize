# Preserved v1 preflight failure

The v1 command verified all sealed source hashes, then failed before loading the
checkpoint or producing `weight-screen.json`:

`ModuleNotFoundError: No module named 'clark_air'`

Cause: the command set `PYTHONPATH=experiments/browser_swin2sr`, which exposed
the experiment modules but omitted the repository root containing `clark_air`.
The v1 manifest is preserved with SHA-256
`6826063391634b28c0468533f15352d94047f9d0a5ab62fe5693c8378bea178d`.
No model, training data, development data, or final-heldout data was read by the
failed analyzer process.
