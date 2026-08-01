# Pre-materialization failure

The v1 audit stopped before writing a materialized model or browser artifact.
Its frozen materializer reconstructed the historical MPS hard projection from
the continuous checkpoint on CPU and correctly rejected the result because
264 tensors differed from the exact evaluated state. The differences were
small (`<= 5.96e-8` in the sampled report) but violated the tensor-identity
contract.

The v1 manifest and source hash remain immutable evidence of that failure.
They must not be reused or amended. A v2 audit instead treats the exact
evaluated dense state as authoritative and reconstructs only the reversible
Hadamard runtime basis from it.
