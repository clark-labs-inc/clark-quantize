# Qwen3.5 Clark Air Real Multimodal QAT Experiment

Goal: replace the fixed10 survival-cache experiment with a real, sharded teacher-behavior corpus for all-2D ternary QAT.

## Target

- Build approximately 100k teacher-labeled records.
- Mix text, image, and video-capable data.
- Include explicit reasoning traces with Qwen-compatible `<think>` tokens.
- Train all 2D tensors with hard-forward ternary g128 and continuous norms.
- Preserve generation behavior, eventually testing long outputs up to 32k new tokens.

## Current Validated Cache Schema

Format: `qwen_agentworld_teacher_behavior_v1`

Each row stores:

- `input_ids`
- `attention_mask`
- optional Qwen3-VL multimodal tensors:
  - `mm_token_type_ids`
  - `pixel_values`
  - `image_grid_thw`
  - `pixel_values_videos`
  - `video_grid_thw`
  - `second_per_grid_ts`
- teacher labels:
  - `top_values`
  - `top_indices`
- metadata:
  - `task`
  - `id`
  - `tokens`
  - `keep`
  - `source_meta`

The trainer consumes these through `row_to_inputs`, so text and image rows now use the same QAT path.

Image/video processor sizing:

- Default builder image size kwargs:
  - `longest_edge=200704`
  - `shortest_edge=4096`
- Default builder video size kwargs:
  - `longest_edge=200704`
  - `shortest_edge=4096`
- This avoids Qwen processor failures where `truncation=True` cuts image/video
  placeholder tokens before the processor can match them to visual grids.
- A previously failing multimodal OpenR1 row at offset 128 now validates as:
  - `input_ids`: `(330,)`
  - `pixel_values`: `(704, 1536)`
  - `image_grid_thw`: `(1, 3)`
- Production before/after:
  - pre-fix shard `006`, source group `textvqa,visual7w,websight`:
    736 rows, 96 failures
  - post-fix shard `010`, same source group: 832 rows, 0 failures

## Sources

### Text

- `CohereLabs/aya_dataset`
  - Multilingual prompt/response data.
  - Used in shard 000.

- `open-r1/OpenR1-Math-220k`
  - Math reasoning traces.
  - Contains `generations` with explicit `<think>...</think>` traces.
  - Validated in `pilot_reason_mm_16.pt`.

- `HuggingFaceFW/fineweb-2`
  - Multilingual web text.
  - Planned configs:
    - `spa_Latn`
    - `fra_Latn`
    - `deu_Latn`
    - `arb_Arab`
    - `hin_Deva`
    - `zho_Hans`
    - `rus_Cyrl`
    - `jpn_Jpan`
    - `kor_Hang`

### Image

- `HuggingFaceM4/the_cauldron`
  - Validated with Qwen3-VL processor.
  - Initial configs:
    - `ai2d`
    - `chartqa`
    - `docvqa`
    - `infographic_vqa`
    - `ocrvqa`
    - `scienceqa`
    - `textvqa`
    - `vqav2`
  - Later add:
    - `visual7w`
    - `websight`
    - `plotqa`
    - `st_vqa`
    - `screen2words`

- `lmms-lab/multimodal-open-r1-8k-verified`
  - Image reasoning rows.
  - Contains image + problem + `<think>...</think><answer>...</answer>` solution.
  - Validated in `pilot_reason_mm_16.pt`.

### Video

- `ShareGPTVideo/train_video_and_instruction`
  - Validated with the Qwen3-VL `AutoProcessor` using real frame groups from
    `video_instruction/train/sft/video_caption_300k.jsonl`.
  - The processor emits:
    - `mm_token_type_ids`
    - `pixel_values_videos`
    - `video_grid_thw`
  - First validated row:
    - video id: `v_--0edUL8zmA-Scene-001`
    - sampled frames: 8
    - `input_ids`: `(362,)`
    - `pixel_values_videos`: `(640, 1536)` stored as FP16 in cache rows
    - `video_grid_thw`: `(1, 3)`
  - This path is opt-in via `--sharegptvideo-samples`; it is not part of the
    current text/image/reasoning wave yet.

Additional candidate sources:

- `lmms-lab/LLaVA-Video-178K`
  - Rich QA JSON plus tarred videos.
  - More complete, but requires a JSON/video archive joiner.
- `lmms-lab/VideoChatGPT`
  - Small QA parquet plus `videos.zip`.
  - Useful for smoke tests but not enough scale by itself.

## Sharding

Do not create one 100k `.pt` file. Use 500-2,000 row shards.

Current shard shape:

- 512 Aya multilingual text rows
- 512 target image rows from Cauldron
- `max_length=1024`
- `last_n=96`
- `top_k=256`

Shard 000 result:

- 937 valid rows
- 87 skipped external rows
- 512 text rows
- 425 image rows
- 1.8 GB on disk

Reasoning pilot:

- `pilot_reason_mm_16.pt`
- 16 valid rows
- 4 Aya multilingual text rows
- 4 OpenR1 text reasoning rows with `<think>`
- 4 multimodal OpenR1 image reasoning rows with `<think>`
- 4 Cauldron image rows

Continuous builder wave:

- Robust script: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/build_continuous_reason_mm_shards_robust.sh`
- Per shard target:
  - 256 Aya multilingual rows
  - 256 OpenR1 text reasoning rows
  - 128 multimodal OpenR1 image reasoning rows
  - rotated Cauldron config groups to reduce HF 504 failures
- Active wave: shards `003..080`
- Queued wave: shards `081..140`, launched by
  `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/build_continuous_reason_mm_shards_wave2_watcher.pid`
- As of the video-validation update:
  - reason shards complete: `001`, `002`, `003`
  - reason rows complete: 2,646
  - legacy shard rows: 937
  - usable rows: 3,583
  - reason rows with explicit `<think>` traces: 1,105

Projected 100k size is now feasible on the cleaned A100 disk, but still large.
Build/train in windows and archive or remove old shards after checkpoint
promotion if disk pressure returns.

## Commands

Shard 000:

```bash
/workspace/venv/bin/python experiments/qwen_agentworld/collect_multimodal_teacher_cache.py \
  --model-dir /workspace/models/Qwen3.5-9B \
  --out /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/mm_shard_000_text512_image512_len1024_last96_top256.pt \
  --max-length 1024 \
  --last-n 96 \
  --top-k 256 \
  --aya-samples 512 \
  --cauldron-configs ai2d,chartqa,docvqa,infographic_vqa,ocrvqa,scienceqa,textvqa,vqav2 \
  --cauldron-per-config 64 \
  --max-rows 1024
```

Shard N offsets:

```bash
AYA_OFFSET=$((N * 512))
CAULDRON_OFFSET=$((N * 64))
/workspace/venv/bin/python experiments/qwen_agentworld/collect_multimodal_teacher_cache.py \
  --aya-offset "$AYA_OFFSET" \
  --cauldron-offset "$CAULDRON_OFFSET" \
  ...
```

Reasoning/image shard wave:

```bash
nohup env START_SHARD=1 END_SHARD=20 \
  /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/build_continuous_reason_mm_shards.sh \
  > /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/build_continuous_reason_mm_shards.log 2>&1 &
```

ShareGPTVideo video shard pilot:

```bash
/workspace/venv/bin/python experiments/qwen_agentworld/collect_multimodal_teacher_cache.py \
  --model-dir /workspace/models/Qwen3.5-9B \
  --out /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/mm_video_pilot_sharegptvideo_016_len1024_last96_top256.pt \
  --max-length 1024 \
  --last-n 96 \
  --top-k 256 \
  --aya-samples 0 \
  --openr1-samples 0 \
  --mm-openr1-samples 0 \
  --fineweb-per-config 0 \
  --cauldron-per-config 0 \
  --sharegptvideo-samples 16 \
  --sharegptvideo-max-frames 8 \
  --max-rows 16
```

Queued ShareGPTVideo video shard wave:

```bash
nohup env START_SHARD=0 END_SHARD=15 \
  /workspace/clark-quantize/experiments/qwen_agentworld/build_sharegptvideo_teacher_shards.sh \
  > /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/build_sharegptvideo_teacher_shards.log 2>&1 &
```

The queued production wave should wait until the text/image/reasoning builder
finishes, so only one teacher-cache GPU job runs at a time.

Metadata-only cache index:

```bash
/workspace/venv/bin/python experiments/qwen_agentworld/summarize_teacher_cache_shards.py \
  "/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/mm_shard_*.pt.json" \
  "/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/mm_reason_shard_*.pt.json" \
  "/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/mm_video_shard_*.pt.json" \
  --out /workspace/agentworld_runs/qwen35_multimodal_cache_20260630/cache_index_latest.json
```

Latest validated index at the time this section was added:

- shards: 7
- rows: 5,902
- modalities: text 3,584, image 2,318
- explicit reasoning/`<think>` rows: 2,176
- bad shards: 0
- missing required metadata fields: 0
- projected 100k cache size at current density: about 163 GB

An A100-side refresher now updates this report every 300 seconds:

- script: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/refresh_cache_index_loop.sh`
- PID file: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/cache_index_refresher.pid`
- log: `/workspace/agentworld_runs/qwen35_multimodal_cache_20260630/cache_index_refresher.log`

First refresher output:

- shards: 8
- rows: 6,647
- modalities: text 4,096, image 2,551
- explicit reasoning/`<think>` rows: 2,532
- bad shards: 0
- projected 100k cache size at current density: about 161 GB

Training:

```bash
/workspace/clark-quantize/experiments/qwen_agentworld/launch_real_mm_all2d_qat_after_cache.sh
```

The launcher is guarded:

- refreshes `cache_index_latest.json`
- refuses to train with any bad shards
- refuses to train below `MIN_ROWS=100000`
- uses all cache families:
  - `mm_shard_*.pt`
  - `mm_reason_shard_*.pt`
  - `mm_video_shard_*.pt`
- phase 1: 5k-step alpha ramp, all-2D ternary g128, mirror SGD
- phase 2: 50k-step hard ternary continuation from phase 1
- post-phase-2 long-generation gate is enabled by default:
  - rows: fixed10
  - lengths: 4,096 / 8,192 / 16,384 / 32,768 new tokens
  - output: `$RUN_ROOT/long_generation_gate/summary.json`

Dry guard receipt:

```text
cache rows=9904 bad_shards=0 min_rows=999999
refusing to train: cache rows below MIN_ROWS
dry_guard_exit=3
```

Second dry guard after adding the long-generation hook:

```text
cache rows=10736 bad_shards=0 min_rows=999999
refusing to train: cache rows below MIN_ROWS
dry_guard_exit=3
```

## Long Generation Gate

Do not judge success by cache KL only.

Add long-generation gates after early cache KL becomes non-catastrophic:

1. Short fixed10 survival.
2. 4k new-token generation.
3. 8k new-token generation.
4. 16k new-token generation.
5. 32k new-token generation.

For every gate, track:

- valid output format
- repeated `<think>`
- newline collapse
- protocol tags
- alnum fraction
- completion length
- whether output remains semantically coherent

## Current Run

Run directory:

`/workspace/agentworld_runs/qwen35_real_mm_qat_20260630`

Tag:

`qwen35_real_mm_t2_mirror_sharded_v1`

This run starts from clean all-2D ternary PTQ over the Qwen3.5 base model, not from the old fixed10 narrow-cache checkpoint.
