# Qwen3.8-27B canonical CQ3 container v1

Status: implemented and synthetic-roundtrip tested. No full 27B artifact,
quality, direct-kernel, or breakthrough claim follows from this format alone.

## Boundary

The `.cq3` file is the portable storage artifact. It binds the exact
`Qwen/Qwen3.8-27B` revision, storage strategy, complete tensor inventory,
parameter denominator, auxiliary-file ledger, and every payload section. An
ISA-specific GGUF compute image is a derived cache and is not allowed to replace
the canonical byte ledger.

The fixed 128-byte little-endian header records the payload and directory
envelopes, section count, complete parameter count, and directory SHA-256. The
payload begins at byte 4096. Every section begins on a 64-byte boundary and has
an independent SHA-256. A canonical JSON directory follows the payload. The
32-byte footer is a root digest over the final header and directory, which bind
all ordered section hashes. Atomic publication uses a same-directory temporary
file, file `fsync`, rename, and directory `fsync`.

## Ternary tensor rule

Rank-2-or-higher tensors flatten dimensions after the first into one row. Each
row resets the group-128 scale and base-3 codeword boundary. Full groups are
128 trits in exactly 203 bits; the final group uses
`ceil(remainder * log2(3))` frozen bits. Row codewords are concatenated as a bit
stream without byte padding between rows. The tensor section alone is rounded
to a byte, then its following section is aligned to 64 bytes.

Each tensor has a separate packed 4-bit scale-index section and a 16-entry
little-endian FP16 codebook section. Rank-1 tensors retain their authenticated
source dtype and raw bytes. The directory tensor records reference all sections
and must sum exactly to the header's whole-checkpoint parameter count.

## Integrity and claims

Opening a container verifies header invariants, directory/root digests,
canonical JSON, ordered non-overlapping section bounds, zero alignment padding,
tensor-to-section references, the exact parameter ledger, and metadata byte
arithmetic. `verify_all()` additionally streams and hashes every payload.

This closes the atomic container/inventory contract only. Promotion still
requires an actual complete 27B export whose measured bytes reconcile with the
1.624562 projection, full scale-codebook evidence, frozen capability tests, and
paired CPU runtime measurements.
