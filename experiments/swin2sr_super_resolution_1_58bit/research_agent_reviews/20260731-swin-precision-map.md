# Swin2SR deployed precision map

- Source task: `/root/swin_precision_map`
- Source session: `019fb948-faba-7822-87b9-c01cd256ae22`
- Captured: 2026-08-01
- Status: archival research lead; verify claims against primary evidence before use


Canonical evidence: [Swin2SR artifact index](../ARTIFACT_INDEX.md).

## Report

Read-only findings from the exact local Swin2SR artifacts:

### Precision structure

The selected research checkpoint is:

- `experiments/browser_swin2sr/surface_group_recovery_20260731/refined/run/checkpoint-final.pt`
- SHA-256 `0b155f9ada8b747544ba1abb83596e160927fdc91a4c33bf36879ff3f77ce00f`
- Selected by [surface-group-recovery-decision.json](../../browser_swin2sr/surface_group_recovery_20260731/surface-group-recovery-decision.json):22-39,76-85.

The exact teacher is `caidas/swin2SR-lightweight-x2-64`, revision `07e8e61492ce6daf4f34c651622960faf110c17f`, pinned in [train_swin2sr_qat.py](../../browser_swin2sr/train_swin2sr_qat.py):52-53.

The refined training checkpoint contains 484 FP32 tensors. Deployment is mixed:

| Deployment class | Tensors | Values |
|---|---:|---:|
| Ternary weights | 204 | 977,604 |
| Raw FP16 | 280 | 27,744 |

Thus 97.2404% of parameters are ternary. The raw FP16 remainder consists of:

- 50 `other_weight` tensors: 3,000 values
- 230 non-weight tensors: 24,744 values—biases, LayerNorm weights/biases, attention logit scales, etc.

Evidence: [model-surface-ledger.json](../../browser_swin2sr/model-surface-ledger.json):5-30 and category implementation in [model_surface_ledger.py](../../browser_swin2sr/model_surface_ledger.py):32-66.

The exporter packs selected weights as ternary but converts every unselected floating tensor to FP16: [export_browser_artifact.py](../../browser_swin2sr/export_browser_artifact.py):184-279. Ternary codes use 20 trits per `uint32`, or approximately 1.6 stored bits/trit: [packing.py](../../../clark_air/quant/packing.py):11-27.

The important Bonsai-like nuance is that the retained high-precision information is mostly in the per-group FP16 scales—not full dense weights. Current FP16-scale storage is:

- 1.60094 packed trit bits/quantized weight
- 0.58183 FP16-scale bits/quantized weight
- 2.18277 combined trunk bits/quantized weight
- 2.56408 whole-model data bits/parameter after raw FP16 tensors

Sensitive surfaces use smaller groups, retaining substantially more scale information. The exact selected deployment distribution I reproduced from the canonical materialization was:

| Surface representation | Tensors | Values | Zero trits |
|---|---:|---:|---:|
| Core linear, group 60 | 143 | 687,600 | 213,532 |
| Core linear, group 3 | 1 | 3,600 | 890 |
| Image I/O, group 2 | 2 | 8,100 | 2,078 |
| Position bias, group 60 | 46 | 96,256 | 34,732 |
| Position bias, group 2 | 2 | 2,048 | 493 |
| Residual conv, group 9 | 5 | 162,000 | 77,266 |
| Residual conv, group 60 | 5 | 18,000 | 7,266 |
| Raw FP16 | 280 | 27,744 | — |

A group-2 tensor pays roughly eight FP16 scale bits per weight in addition to ternary codes. So the first and final convolutions are already acting like high-precision islands through their scale density even though their actual weight codes remain ternary. The five selected group overrides are recorded in [materialization.json](../../browser_swin2sr/surface_group_recovery_20260731/refined/canonical/materialization.json):19-25; exact state identities are at lines 1552-1564.

### Q1/Q2/Q4 meaning

The ongoing Q1/Q2/Q4 work quantizes the **scale indices**, not the ternary weight codes. Each tensor still carries FP16 codebook centers:

- FP16-rounded learned centers: [scale_codebook_qat.py](../../browser_swin2sr/scale_codebook_qat.py):95-118
- Category-specific scale schedule: lines 14-19 and 160-178
- Browser storage of FP16 centers plus packed indices: [export_browser_artifact.py](../../browser_swin2sr/export_browser_artifact.py):193-233

The strongest static mixed map found was:

- core scales Q1
- image-I/O scales Q4
- position-bias scales Q2
- residual-convolution scales Q2

It reached 1.67970 trunk bits/weight, but without QAT it lost 1.93448 dB versus the FP16-scale refined checkpoint. It is therefore a bridge curriculum, not a selected replacement. Evidence: [mixed-scale result](../../browser_swin2sr/mixed_scale_map_probe_20260731/result.json):647-720.

### Existing sensitivity analyses

The reusable local analyses are:

- Teacher category/count/storage ledger: [model_surface_ledger.py](../../browser_swin2sr/model_surface_ledger.py):32-140
- Per-surface ternary code churn, sparsity, sign inheritance and continuous-master movement: [analyze_surface_signatures.py](../../browser_swin2sr/analyze_surface_signatures.py):33-163
- One-at-a-time restoration from ternary to the trained continuous master: [analyze_surface_restore_sensitivity.py](../../browser_swin2sr/analyze_surface_restore_sensitivity.py):169-281
- Task-gradient estimate for changing scale group size: [analyze_gradient_group_sensitivity.py](../../browser_swin2sr/analyze_gradient_group_sensitivity.py):82-268
- Exact hard-ternary code/scale recovery: [rewrite_onnx_ternary.py](../../browser_swin2sr/rewrite_onnx_ternary.py):21-50
- Historical full-precision-island evaluator: [evaluate_restoration_islands.py](../../browser_swin2sr/evaluate_restoration_islands.py):46-90,168-241

No existing script combines teacher distributions, trained continuous-master distributions, exact deployed trits/scales, and effective precision into one receipt.

### Smallest no-write combined map

This command only loads tensors and prints JSON; it performs no inference or training:

```bash
cd /Users/stan/Documents/git/clark-quantize
PYTHONPATH=experiments/browser_swin2sr:. python3 - <<'PY'
import json
import numpy as np
import torch
from pathlib import Path

from evaluate_hard_started_checkpoints import load_checkpoint_student
from materialize_surface_states import load_materialized_state
from rewrite_onnx_ternary import recover_ternary_groups
from surface_model_state import pretrained

checkpoint = Path(
    "experiments/browser_swin2sr/"
    "surface_group_recovery_20260731/refined/run/checkpoint-final.pt"
)
manifest_path = checkpoint.parents[1] / "canonical/materialization.json"

manifest, runtime = load_materialized_state(manifest_path, "runtime_state")
student, surfaces, payload = load_checkpoint_student(checkpoint, "cpu")
teacher = pretrained("cpu").state_dict()
surface_by_name = {f"{s.name}.weight": s for s in surfaces}

def distribution(value):
    x = value.detach().float().reshape(-1)
    absolute = x.abs()
    q = torch.quantile(
        absolute,
        torch.tensor([0.50, 0.90, 0.99], dtype=absolute.dtype),
    )
    return {
        "values": x.numel(),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "abs_p50": float(q[0]),
        "abs_p90": float(q[1]),
        "abs_p99": float(q[2]),
        "zero_fraction": float((x == 0).float().mean()),
    }

rows = []
for name, teacher_tensor in sorted(teacher.items()):
    surface = surface_by_name.get(name)
    current = runtime[name]
    row = {
        "name": name,
        "shape": list(teacher_tensor.shape),
        "teacher_precision": "fp32",
        "teacher": distribution(teacher_tensor),
    }
    if surface is None:
        row.update({
            "category": "raw_correction_or_other_weight",
            "refined_master_precision": "fp32",
            "deployment_precision": "fp16",
            "refined_master": distribution(current),
            "deployment": distribution(current.half()),
        })
    else:
        master = surface.module.parametrizations.weight.original.detach().cpu()
        spec = manifest["surfaces"][name]
        codes, scales, error = recover_ternary_groups(
            current.numpy(), int(spec["group_size"])
        )
        row.update({
            "category": spec["category"],
            "refined_master_precision": "fp32",
            "deployment_precision": "packed_ternary_plus_fp16_scale",
            "group_size": int(spec["group_size"]),
            "quantizer": spec["method"],
            "rotation": spec["rotation"],
            "refined_master": distribution(master),
            "deployment": distribution(current),
            "trits": {
                "-1": int((codes == -1).sum()),
                "0": int((codes == 0).sum()),
                "+1": int((codes == 1).sum()),
            },
            "scales": {
                "count": int(scales.size),
                "minimum": float(scales.min()),
                "maximum": float(scales.max()),
                "mean": float(scales.mean()),
                "p50": float(np.quantile(scales, 0.50)),
                "p90": float(np.quantile(scales, 0.90)),
                "p99": float(np.quantile(scales, 0.99)),
                "recovery_max_error": error,
            },
        })
    rows.append(row)

print(json.dumps({
    "teacher_model": "caidas/swin2SR-lightweight-x2-64",
    "teacher_revision": "07e8e61492ce6daf4f34c651622960faf110c17f",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "surface_step": payload["surface_step"],
    "tensors": rows,
}, indent=2))
PY
```

The immediate research implication is to treat tensor precision as three independent knobs:

1. Ternary code versus raw FP16 island.
2. Per-tensor/group scale density.
3. Scale-codebook index precision while retaining FP16 centers.

That is more faithful to the local evidence than trying to force every tensor and every scale to the same nominal bit width.
