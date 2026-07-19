"""Build broad text/image/video teacher-label caches for Qwen3-VL/Qwen3.5 QAT.

The output intentionally uses qwen_agentworld_teacher_behavior_v1 so existing
all-2D QAT trainers can consume it. Rows may include Qwen processor multimodal
tensors such as pixel_values, image_grid_thw, pixel_values_videos, and
video_grid_thw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import chain
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_onpolicy_survival_cache import load_model, token_ids_for_strings


TEXT_KEYS = ("text", "context", "prompt", "question", "input", "inputs")
TARGET_KEYS = ("target", "targets", "answer", "response", "output", "completion")
PROCESSOR_TENSOR_KEYS = (
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)
BATCHED_ROW_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}
PIXEL_KEYS = {"pixel_values", "pixel_values_videos"}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def aya_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = first_text(row, ("inputs", "input", "prompt", "question"))
    target = first_text(row, ("targets", "target", "answer", "response"))
    if not prompt:
        raise ValueError("Aya row has no prompt text")
    if target:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    return [{"role": "user", "content": prompt}]


def generic_text_messages(row: dict[str, Any]) -> list[dict[str, str]] | str:
    prompt = first_text(row, TEXT_KEYS)
    target = first_text(row, TARGET_KEYS)
    if not prompt:
        messages = row.get("messages")
        if isinstance(messages, list):
            return [
                {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                for msg in messages
                if isinstance(msg, dict) and msg.get("content")
            ]
        raise ValueError(f"text row has no usable text: keys={sorted(row)}")
    if target:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    return prompt


def cauldron_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    images = row.get("images") or []
    texts = row.get("texts") or []
    if not images or not texts:
        raise ValueError("Cauldron row missing images/texts")
    turn = texts[0]
    user = turn.get("user") if isinstance(turn, dict) else None
    assistant = turn.get("assistant") if isinstance(turn, dict) else None
    if not user or not assistant:
        raise ValueError("Cauldron row missing user/assistant text")
    image = images[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(user)},
            ],
        },
        {"role": "assistant", "content": str(assistant)},
    ]
    return messages, [image]


def openr1_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    problem = first_text(row, ("problem", "question", "prompt"))
    if not problem:
        raise ValueError("OpenR1 row has no problem")
    generations = row.get("generations")
    answer = None
    if isinstance(generations, list):
        complete = row.get("is_reasoning_complete")
        for index, generation in enumerate(generations):
            if not isinstance(generation, str) or not generation.strip():
                continue
            if isinstance(complete, list) and index < len(complete) and not complete[index]:
                continue
            answer = generation.strip()
            break
        if answer is None:
            for generation in generations:
                if isinstance(generation, str) and generation.strip():
                    answer = generation.strip()
                    break
    if answer is None:
        answer = first_text(row, ("solution", "answer", "answer_content"))
    if not answer:
        raise ValueError("OpenR1 row has no reasoning answer")
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": answer},
    ]


def mm_openr1_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    image = row.get("image")
    problem = first_text(row, ("problem", "original_question", "question"))
    solution = first_text(row, ("solution", "original_answer", "answer"))
    if image is None or not problem or not solution:
        raise ValueError("multimodal OpenR1 row missing image/problem/solution")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem},
            ],
        },
        {"role": "assistant", "content": solution},
    ]
    return messages, [image]


def sharegpt_video_id_from_key(key: str) -> str:
    parts = key.strip().lstrip("./").split("/")
    if not parts or not parts[0]:
        raise ValueError(f"bad ShareGPTVideo frame key: {key!r}")
    return parts[0]


def sharegpt_video_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    frames = row.get("frames")
    conversations = row.get("conversations")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ShareGPTVideo row missing frames")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("ShareGPTVideo row missing conversations")
    user_text = None
    assistant_text = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", turn.get("role", ""))).lower()
        value = str(turn.get("value", turn.get("content", ""))).strip()
        if not value:
            continue
        if speaker in {"human", "user"} and user_text is None:
            user_text = value.replace("<video>", "").strip()
        if speaker in {"gpt", "assistant"} and assistant_text is None:
            assistant_text = value
    if not assistant_text:
        raise ValueError("ShareGPTVideo row missing assistant answer")
    if not user_text:
        user_text = "Describe the video."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]
    return messages, [frames]


def tensor_payload(processed: Any) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for key in PROCESSOR_TENSOR_KEYS:
        value = processed.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"} and value.ndim > 1:
            value = value[0]
        if key in PIXEL_KEYS:
            value = value.to(torch.float16)
        payload[key] = value.detach().cpu()
    return payload


def model_inputs_from_payload(payload: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    model_inputs: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        tensor = value.to(device=device)
        if key in BATCHED_ROW_KEYS and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        model_inputs[key] = tensor
    return model_inputs


def build_inputs(
    processor: Any,
    sample: dict[str, Any],
    *,
    source_kind: str,
    max_length: int,
    image_size: dict[str, int] | None,
    video_size: dict[str, int] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if source_kind == "aya":
        messages = aya_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(text=[text], return_tensors="pt", truncation=True, max_length=max_length)
        return tensor_payload(processed), {"modality": "text", "language": sample.get("language_code") or sample.get("language")}
    if source_kind == "text":
        content = generic_text_messages(sample)
        if isinstance(content, list):
            text = processor.apply_chat_template(content, tokenize=False, add_generation_prompt=False)
        else:
            text = str(content)
        processed = processor(text=[text], return_tensors="pt", truncation=True, max_length=max_length)
        return tensor_payload(processed), {"modality": "text"}
    if source_kind == "cauldron":
        messages, images = cauldron_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            **({"images_kwargs": {"size": image_size}} if image_size else {}),
        )
        return tensor_payload(processed), {"modality": "image"}
    if source_kind == "openr1":
        messages = openr1_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(text=[text], return_tensors="pt", truncation=True, max_length=max_length)
        return tensor_payload(processed), {"modality": "text", "reasoning": True, "think_tokens": "<think>" in text}
    if source_kind == "mm_openr1":
        messages, images = mm_openr1_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            **({"images_kwargs": {"size": image_size}} if image_size else {}),
        )
        return tensor_payload(processed), {"modality": "image", "reasoning": True, "think_tokens": "<think>" in text}
    if source_kind == "sharegpt_video":
        messages, videos = sharegpt_video_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        processed = processor(
            text=[text],
            videos=videos,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            **({"videos_kwargs": {"size": video_size}} if video_size else {}),
        )
        return tensor_payload(processed), {"modality": "video", "frames": len(sample.get("frames", []))}
    raise ValueError(f"unknown source kind: {source_kind}")


def maybe_skip(dataset: Any, offset: int) -> Any:
    if offset <= 0:
        return dataset
    if hasattr(dataset, "skip"):
        return dataset.skip(offset)
    return dataset


def stream_aya(limit: int, offset: int) -> Iterable[tuple[str, dict[str, Any], str]]:
    if limit <= 0:
        return
    dataset = load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    dataset = maybe_skip(dataset, offset)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield "aya", row, f"aya:{offset + index}"


def stream_fineweb(configs: list[str], per_config: int, offset: int) -> Iterable[tuple[str, dict[str, Any], str]]:
    if per_config <= 0:
        return
    for config in configs:
        try:
            dataset = load_dataset("HuggingFaceFW/fineweb-2", config, split="train", streaming=True)
        except Exception as exc:  # noqa: BLE001 - external configs drift.
            print(
                json.dumps(
                    {
                        "skip_fineweb_config": config,
                        "error": type(exc).__name__,
                        "message": str(exc)[:240],
                    }
                ),
                flush=True,
            )
            continue
        dataset = maybe_skip(dataset, offset)
        for index, row in enumerate(dataset):
            if index >= per_config:
                break
            yield "text", row, f"fineweb2:{config}:{offset + index}"


def stream_cauldron(configs: list[str], per_config: int, offset: int) -> Iterable[tuple[str, dict[str, Any], str]]:
    if per_config <= 0:
        return
    for config in configs:
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config, split="train", streaming=True)
        dataset = maybe_skip(dataset, offset)
        for index, row in enumerate(dataset):
            if index >= per_config:
                break
            yield "cauldron", row, f"cauldron:{config}:{offset + index}"


def stream_openr1(limit: int, offset: int) -> Iterable[tuple[str, dict[str, Any], str]]:
    if limit <= 0:
        return
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    dataset = maybe_skip(dataset, offset)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield "openr1", row, f"openr1_math:{offset + index}"


def stream_mm_openr1(limit: int, offset: int) -> Iterable[tuple[str, dict[str, Any], str]]:
    if limit <= 0:
        return
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train", streaming=True)
    dataset = maybe_skip(dataset, offset)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield "mm_openr1", row, f"mm_openr1:{offset + index}"


def read_sharegptvideo_targets(
    metadata_file: str,
    *,
    offset: int,
    limit: int,
    target_multiplier: int,
) -> dict[str, dict[str, Any]]:
    if limit <= 0:
        return {}
    path = hf_hub_download(
        "ShareGPTVideo/train_video_and_instruction",
        metadata_file,
        repo_type="dataset",
    )
    wanted = max(limit, limit * max(1, target_multiplier))
    targets: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if len(targets) >= wanted:
                break
            row = json.loads(line)
            video_id = str(row.get("video") or row.get("id") or "").strip()
            if not video_id or video_id in targets:
                continue
            targets[video_id] = row
    return targets


def sample_video_frames(frames: list[Any], max_frames: int) -> list[Any]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return [frames[0]]
    step = (len(frames) - 1) / float(max_frames - 1)
    return [frames[round(i * step)] for i in range(max_frames)]


def stream_sharegptvideo(
    limit: int,
    offset: int,
    *,
    metadata_file: str,
    max_frames: int,
    target_multiplier: int,
) -> Iterable[tuple[str, dict[str, Any], str]]:
    if limit <= 0:
        return
    targets = read_sharegptvideo_targets(
        metadata_file,
        offset=offset,
        limit=limit,
        target_multiplier=target_multiplier,
    )
    if not targets:
        return
    dataset = load_dataset("ShareGPTVideo/train_video_and_instruction", split="train", streaming=True)
    current_video: str | None = None
    current_frames: list[Any] = []
    yielded = 0

    def flush() -> tuple[str, dict[str, Any], str] | None:
        nonlocal yielded
        if current_video is None or yielded >= limit:
            return None
        target = targets.get(current_video)
        if target is None or not current_frames:
            return None
        row = dict(target)
        row["frames"] = sample_video_frames(current_frames, max_frames)
        yielded += 1
        return "sharegpt_video", row, f"sharegptvideo:{metadata_file}:{offset}:{current_video}"

    for frame_row in dataset:
        key = str(frame_row.get("__key__", ""))
        if not key:
            continue
        video_id = sharegpt_video_id_from_key(key)
        if current_video is None:
            current_video = video_id
        if video_id != current_video:
            item = flush()
            if item is not None:
                yield item
                if yielded >= limit:
                    return
            current_video = video_id
            current_frames = []
        if video_id in targets:
            image = frame_row.get("jpeg")
            if image is not None:
                current_frames.append(image.convert("RGB") if hasattr(image, "convert") else image)
    item = flush()
    if item is not None:
        yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--image-longest-edge", type=int, default=200704)
    parser.add_argument("--image-shortest-edge", type=int, default=4096)
    parser.add_argument("--video-longest-edge", type=int, default=200704)
    parser.add_argument("--video-shortest-edge", type=int, default=4096)
    parser.add_argument("--last-n", type=int, default=96)
    parser.add_argument("--dense-positions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--aya-samples", type=int, default=512)
    parser.add_argument("--aya-offset", type=int, default=0)
    parser.add_argument("--openr1-samples", type=int, default=0)
    parser.add_argument("--openr1-offset", type=int, default=0)
    parser.add_argument("--mm-openr1-samples", type=int, default=0)
    parser.add_argument("--mm-openr1-offset", type=int, default=0)
    parser.add_argument("--sharegptvideo-samples", type=int, default=0)
    parser.add_argument("--sharegptvideo-offset", type=int, default=0)
    parser.add_argument(
        "--sharegptvideo-metadata",
        default="video_instruction/train/sft/video_caption_300k.jsonl",
    )
    parser.add_argument("--sharegptvideo-max-frames", type=int, default=8)
    parser.add_argument("--sharegptvideo-target-multiplier", type=int, default=5)
    parser.add_argument("--fineweb-configs", default="spa_Latn,fra_Latn,deu_Latn,arb_Arab,hin_Deva,cmn_Hani,rus_Cyrl,jpn_Jpan,kor_Hang")
    parser.add_argument("--fineweb-per-config", type=int, default=0)
    parser.add_argument("--fineweb-offset", type=int, default=0)
    parser.add_argument(
        "--cauldron-configs",
        default="ai2d,chartqa,docvqa,infographic_vqa,ocrvqa,scienceqa,textvqa,vqav2,visual7w,websight",
    )
    parser.add_argument("--cauldron-per-config", type=int, default=64)
    parser.add_argument("--cauldron-offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=1152)
    parser.add_argument("--fast-exit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    image_size = {"longest_edge": args.image_longest_edge, "shortest_edge": args.image_shortest_edge}
    video_size = {"longest_edge": args.video_longest_edge, "shortest_edge": args.video_shortest_edge}
    model = load_model(args.model_dir)
    for param in model.parameters():
        param.requires_grad_(False)
    device = next(model.parameters()).device

    sources = chain(
        stream_aya(args.aya_samples, args.aya_offset),
        stream_openr1(args.openr1_samples, args.openr1_offset),
        stream_mm_openr1(args.mm_openr1_samples, args.mm_openr1_offset),
        stream_sharegptvideo(
            args.sharegptvideo_samples,
            args.sharegptvideo_offset,
            metadata_file=args.sharegptvideo_metadata,
            max_frames=args.sharegptvideo_max_frames,
            target_multiplier=args.sharegptvideo_target_multiplier,
        ),
        stream_fineweb(parse_csv(args.fineweb_configs), args.fineweb_per_config, args.fineweb_offset),
        stream_cauldron(parse_csv(args.cauldron_configs), args.cauldron_per_config, args.cauldron_offset),
    )

    out_rows: list[dict[str, Any]] = []
    failures = 0
    for index, (source_kind, sample, source_id) in enumerate(sources):
        if len(out_rows) >= args.max_rows:
            break
        try:
            payload, meta = build_inputs(
                processor,
                sample,
                source_kind=source_kind,
                max_length=args.max_length,
                image_size=image_size,
                video_size=video_size,
            )
            input_ids = payload["input_ids"]
            sequence_length = int(input_ids.shape[-1])
            keep = sequence_length if args.dense_positions or args.last_n <= 0 else min(args.last_n, sequence_length)
            model_inputs = model_inputs_from_payload(payload, device)
            with torch.no_grad():
                outputs = model(**model_inputs, use_cache=False, logits_to_keep=keep)
            logits = outputs.logits[0, -keep:, :].detach().float()
            teacher_logsumexp = torch.logsumexp(logits, dim=-1)
            top_values, top_indices = torch.topk(logits, k=min(args.top_k, logits.shape[-1]), dim=-1)
            row = {
                "task": source_kind,
                "id": source_id,
                "index": len(out_rows),
                "tokens": int(input_ids.shape[-1]),
                "keep": keep,
                **payload,
                "top_values": top_values.cpu().to(torch.float16),
                "top_indices": top_indices.cpu().to(torch.int32),
                "teacher_logsumexp": teacher_logsumexp.cpu().to(torch.float32),
                "source_meta": meta,
            }
            out_rows.append(row)
            print(
                json.dumps(
                    {
                        "index": row["index"],
                        "source": source_id,
                        "modality": meta.get("modality"),
                        "tokens": row["tokens"],
                        "positions": keep,
                        "top_k": int(top_values.shape[-1]),
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - bad external rows should not kill shard build.
            failures += 1
            print(json.dumps({"skip": source_id, "error": type(exc).__name__, "message": str(exc)[:240]}), flush=True)

    format_token_ids = token_ids_for_strings(
        processor.tokenizer,
        ["<think>", "</think>", "<predicted_observation>", "</predicted_observation>", "\n", "[", "]", "{", "}", '"'],
    )
    collapse_token_ids = token_ids_for_strings(processor.tokenizer, ["<think>", "assistant", "\n\n", "\n"])
    payload = {
        "format": "qwen_agentworld_teacher_behavior_v1",
        "model_dir": str(args.model_dir),
        "data_dir": "hf_stream:CohereLabs/aya_dataset,open-r1/OpenR1-Math-220k,lmms-lab/multimodal-open-r1-8k-verified,ShareGPTVideo/train_video_and_instruction,HuggingFaceFW/fineweb-2,HuggingFaceM4/the_cauldron",
        "max_length": args.max_length,
        "last_n": args.last_n,
        "dense_positions": args.dense_positions,
        "top_k": args.top_k,
        "format_token_ids": format_token_ids,
        "collapse_token_ids": collapse_token_ids,
        "hidden_layers": [],
        "builder": "collect_multimodal_teacher_cache.py",
        "builder_args": vars(args) | {"model_dir": str(args.model_dir), "out": str(args.out)},
        "failures": failures,
        "rows": out_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    meta_path = args.out.with_suffix(args.out.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "rows"
            }
            | {
                "rows": [
                    {
                        "task": row["task"],
                        "id": row["id"],
                        "tokens": row["tokens"],
                        "positions": int(row["top_values"].shape[0]),
                        "top_k": int(row["top_values"].shape[-1]),
                        "meta": row.get("source_meta", {}),
                    }
                    for row in out_rows
                ]
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out": str(args.out), "meta": str(meta_path), "rows": len(out_rows), "failures": failures}, indent=2))
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
