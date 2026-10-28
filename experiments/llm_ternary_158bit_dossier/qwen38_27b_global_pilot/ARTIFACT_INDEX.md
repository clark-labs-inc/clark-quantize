# Artifact Index

Remote authoritative root: `/workspace/quant-state/qwen38-global-pilot-v1`

| Artifact | Required state | Location |
|---|---:|---|
| Frozen capability manifest | complete | `capability_manifest.json` |
| BF16 capability report | complete | `eval_bf16.json` |
| Current base-3 PTQ capability report | complete | `eval_base3_ptq.json` |
| Teacher prompt/cache receipts | complete | `teacher_prompts.json`; cache metadata beside remote cache |
| One-step memory-smoke report | complete | `train_smoke.json`; `train_anchor_smoke.json` |
| Bounded-pilot report | complete | `train_pilot.json` |
| Trained capability report | complete, failed quality | `eval_trained.json` |
| Forgetting comparison | complete, promotion false | `forgetting_comparison.json` |
| Terminal integrity receipt | complete | `terminal_receipt.json` |

All table entries are declarations, not completion claims. A row becomes complete
only after its remote file exists, parses, and has an independently recorded
SHA-256 digest.
