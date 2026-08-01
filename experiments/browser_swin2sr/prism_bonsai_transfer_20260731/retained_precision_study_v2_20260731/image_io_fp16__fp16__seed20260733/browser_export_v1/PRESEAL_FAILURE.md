# Pre-seal failure

This frozen attempt did not produce a promotable browser artifact.

The sealed overlay included the mixed-precision exporter but resolved
`artifact_ledger.py` from an older base-source snapshot. That helper did not
accept the exporter's `offset_bytes` storage field, so export stopped before a
complete `model.json` was written. The evaluated checkpoint and dense state
were not changed.

The corrected attempt must use a new versioned directory whose sealed overlay
includes `artifact_ledger.py` explicitly.
