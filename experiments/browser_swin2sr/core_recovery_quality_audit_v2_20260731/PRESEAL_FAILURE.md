# Artifact-export failure

V2 proved exact evaluated-state tensor identity, then stopped before emitting a
browser artifact. The symmetric ternary exporter rejected the first core group
because this historical CompSRT-style checkpoint has learned affine offsets:
its three code levels are `offset + {-1,0,1} * scale`, not symmetric
`{-1,0,1} * scale`.

No v2 model manifest or binary was completed. The v2 manifest and its frozen
source hashes remain failure evidence and must not be amended or reused. V3
adds an optional FP16 group-offset stream to the packed artifact and native/
WASM runtime while leaving existing symmetric artifacts unchanged.
