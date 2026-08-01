# BackdropKit Swin2SR method map

Research dossier: `experiments/swin2sr_super_resolution_1_58bit/`

Status date: 2026-07-29

Clark Quantize is a scientific reference for this file, not an independent
goal. Every mechanism below is included only when it can improve the tiny
Swin2SR model used by BackdropKit.

| Clark mechanism | Swin2SR status | Product-focused decision |
| --- | --- | --- |
| Continuous master, hard ternary forward, STE | Implemented and under long recovery | Keep as the invariant |
| MSE-optimal support and scale | Implemented; scales are recomputed from masters every forward | Keep as primary control |
| AdamW8bit-style low-state optimizer | Implemented as BonsaiQAT8; matched race active | Promote only on held-out quality or material memory benefit |
| Proximal pull | Isolated in the AdamW-prox arm | Do not assume it helps; current early results trail |
| Periodic scale refresh | Relevant only to learned-scale arms; MSE/absmean already recompute closed-form scales every forward | Add only if ParetoQ learned scales win |
| Muon | One preregistered arm, not presumed | Retain only if paired quality per compute wins |
| Differentiated continuous-island learning rates | Missing from the current frozen-island control | Run frozen, norms, and norms-plus-residual policies after selecting the optimizer |
| Protected image-facing surfaces | Implemented in the current core experiment | Ternarize them last through the staged surface schedule |
| Activation-aware shaping/rotation | Hadamard control active | Promote only with held-out evidence |

The immediate quality question is whether low-rate normalization and residual
compensation closes the exact shipped-q4 gap. `island_policy.py` preregisters
the three policies without changing the active factorial run. The
`norms_and_residual` policy trains 6,000 LayerNorm parameters at 0.10x learning
rate and 180,600 residual-body convolution parameters/biases at 0.05x, while
keeping the 8,172 image-ingress/reconstruction parameters frozen. If this
restores quality, the residual weights then enter the 25–55% ternary schedule;
they are not retained as an unreported high-precision escape hatch.
