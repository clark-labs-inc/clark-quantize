# Failed preflight receipt

This v1 study produced no training results and is not executable authority.

The immutable sealed RSTB ternary-control inspection failed before training:

```text
RuntimeError: ternary control initialization changed the hard weight:
swin2sr.encoder.stages.0.conv: 7.450580596923828e-09
```

The `optimal_mse` projector recomputes its scale when its own hard output is
copied back into the master, so exact bitwise idempotence is too strict. The
observed delta is below one FP32 epsilon. V2 replaces equality with an explicit
four-epsilon bound, records the observed delta and tolerance per tensor, and
must pass sealed inspections for both precision modes before any training.
