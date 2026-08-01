# Video training data journal

## 2026-08-01 — Relevance and rights before raw scale

The largest immediately useful source is not a generic text-to-video dump. The
CC-BY-4.0 `markov-ai/computer-use-large` revision pinned in
`source-manifest.json` contains 48,478 trimmed screen-recording videos and
approximately 12,300 hours of professional software workflows. That is much
closer to BackdropKit's product-demo distribution than cinematic synthetic
video, and it is large enough to supply millions of temporally adjacent frame
windows without treating individual frames as independent images.

The 82-video CC-BY-4.0 Humyn consumer-app collection is smaller but covers
mobile commerce and complete user journeys. Complete workflows must be reserved
before frame extraction so train and validation cannot contain neighboring
frames from the same original recording.

The tempting AI-generated collections are not automatically usable. The live
Seedance 2.0 collection is CC-BY-NC-4.0 and the multi-generator synthetic-video
detector dataset is CC-BY-NC-SA-4.0. AniMINT is CC-BY-NC-ND-4.0. None may
contribute to commercially deployed weights. OpenVid-1M has a CC-BY-4.0 card,
but its own card requires compliance with several underlying datasets and names
an unknown source license, so the full corpus stays quarantined until each item
has a rights-clean provenance decision.

The rights-clean synthetic path is therefore deterministic UI rendering we
control: render seeded product interfaces, charts, terminals, forms, cursors,
scrolling, transitions, and focus motion at high resolution; create the low
resolution side locally; and publish the exact renderer inputs and licenses.
This is also more valuable than generic cinematic video because small text,
one-pixel dividers, icons, charts, and cursor edges are the actual failure
distribution.

## Sealed pairing protocol

1. Split by original video before decoding, never by frame.
2. Decode linear-light RGB high-resolution frames and retain timestamps.
3. Build 3-, 5-, and 7-frame windows with genuine adjacent timing.
4. Generate low-resolution inputs with a recorded mixture of area, Lanczos,
   H.264/AV1/WebM compression, chroma subsampling, ringing, browser resize, and
   fractional scaling; never download an untraceable low-resolution target.
5. Train reconstruction, edge, text-legibility, teacher, temporal-warp, and
   flicker losses. Optical-flow confidence masks occlusions rather than forcing
   hallucinated correspondence.
6. Hold out applications, workflows, languages, and synthetic renderer seeds.
7. Promote only when image PSNR/edge quality and temporal consistency improve
   with no source, application, script, or language regression.

The first bounded ingest should be 1,000 hours from the pinned Markov revision,
all 82 Humyn videos reserved primarily for evaluation, and 100,000 Clark-owned
synthetic UI clips. That is already orders of magnitude more relevant temporal
evidence than downloading a noncommercial Seedance collection.

## 2026-08-01 — Pinned 1,000-hour ingest index

All nine metadata shards in the pinned Markov revision were downloaded and
hashed without downloading the 856 GB of video blobs. The consolidated index
contains 48,477 identified videos and hashes to
`82c6e36508e0f4401bf9a0e8016bb3ebf82af130ab9b698b748c8ef72a56dbf4`.
The source contains 51 identified rows without complete duration metadata and
one row without a category; all 52 are quarantined rather than guessed.

The source-disjoint split assigns 40,342 videos / 9,829.40 hours from AutoCAD,
Blender, Excel, and Photoshop to the training pool; 7,828 Salesforce videos /
2,324.71 hours to validation; and 307 VS Code videos / 127.09 hours to test.
A stable SHA-256 priority selected 4,282 training videos totaling 1,001.51
hours, balanced at approximately 250 hours per training application. This is
the download list for the first bounded ingest; it can be reproduced from
metadata hashes and does not depend on mutable Hub ordering.

## 2026-08-01 — Resolution audit narrows each source's role

Four downloaded Markov samples are 640x360. They are useful for real cursor,
scroll, transition, and compression priors, but they cannot serve as honest
high-resolution targets for 2K-to-4K reconstruction. The extracted 316,210,815
byte Humyn shard contains seven quick-commerce workflows at approximately
382/384 by 848/864 pixels. Those are valuable mobile-motion evaluation clips,
not 4K target frames.

OpenVidHD metadata contains 433,509 rows with strong motion and temporal scores,
but its opaque filenames do not include item-level source-rights fields. The
large archive remains quarantined rather than being treated as production
training data based only on the collection-level card. For the current temporal
student, deterministic motion generated from the rights-clean 206,232-image UI
and multilingual corpus is the only promoted paired-frame source; real videos
remain evaluation or motion-prior inputs until target resolution and provenance
are independently adequate.
