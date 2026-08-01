#!/usr/bin/env python3
"""Generate 100K equally balanced multilingual UI/text canvases.

FLORES-200 supplies professionally translated text in 204 language-script
variants (200 languages plus four alternate scripts). Pango
performs script shaping and font fallback; Pillow draws deterministic UI
geometry. Exactly 500 canvases per variant produces a balanced 102K corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw
from text_renderer import SCRIPT_FONTS, prepare_fontconfig, render_text


FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_HF_SOURCE = "Muennighoff/flores200"
FLORES_HF_REVISION = "9660f10339cfc46a369a4d95fccb301a739c3fa8"

RESOLUTIONS = (
    (320, 240),
    (384, 384),
    (512, 320),
    (640, 480),
    (768, 432),
    (800, 600),
    (1024, 576),
    (1080, 1080),
)

PALETTES = (
    ("#f7f7fb", "#ffffff", "#17171c", "#6d5dfc", "#e7e5f4"),
    ("#0e1117", "#191d26", "#f6f7fb", "#93e635", "#303644"),
    ("#fff9f2", "#ffffff", "#271d16", "#ff5d3b", "#eadfd4"),
    ("#edf7ff", "#ffffff", "#10243b", "#0d8cff", "#d7e8f6"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".cache/clark-quantize/swin2sr-flores200-text-100k-v1",
    )
    parser.add_argument("--per-language", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--split", choices=("dev", "devtest"), default="dev")
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Generate only this FLORES language-script variant. Repeat as needed.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="JSONL manifests whose image hashes must not enter this dataset.",
    )
    parser.add_argument(
        "--reserve-per-language",
        type=int,
        default=0,
        help="Extra candidates rendered per language before exclusion filtering.",
    )
    parser.add_argument(
        "--exclude-text-split",
        choices=("dev", "devtest"),
        help="Exclude raw sentences selected for training from this FLORES split.",
    )
    parser.add_argument(
        "--exclude-text-per-language",
        type=int,
        default=0,
        help="Number of deterministically selected sentences to exclude per language.",
    )
    parser.add_argument(
        "--exclude-text-seed",
        type=int,
        default=20260729,
        help="Selection seed used by the excluded text dataset.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache/clark-quantize/sources",
    )
    return parser.parse_args()


def load_excluded_hashes(manifests: list[Path]) -> set[str]:
    excluded = set()
    for manifest in manifests:
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                digest = json.loads(line).get("sha256")
                if not digest:
                    raise ValueError(f"{manifest} contains a record without sha256")
                excluded.add(digest)
    return excluded


def select_disjoint_records(
    records: list[dict],
    per_language: int,
    excluded_hashes: set[str],
) -> list[dict]:
    selected = []
    seen = set(excluded_hashes)
    by_language: dict[str, list[dict]] = {}
    for record in records:
        by_language.setdefault(record["language"], []).append(record)
    for language, candidates in sorted(by_language.items()):
        accepted = []
        for record in candidates:
            digest = record["sha256"]
            if digest in seen:
                continue
            seen.add(digest)
            accepted.append(record)
            if len(accepted) == per_language:
                break
        if len(accepted) != per_language:
            raise ValueError(
                f"{language} only has {len(accepted)} disjoint images; "
                f"need {per_language}"
            )
        selected.extend(accepted)
    return selected


def sanitize_text(text: str) -> str:
    characters = []
    for character in text:
        codepoint = ord(character)
        if 0x80 <= codepoint <= 0x9F:
            try:
                character = bytes([codepoint]).decode("cp1252")
            except UnicodeDecodeError:
                continue
        if unicodedata.category(character) == "Cc":
            continue
        characters.append(character)
    return unicodedata.normalize("NFC", "".join(characters)).strip()


def fetch_flores(cache: Path, split: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "flores200_dataset.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        urllib.request.urlretrieve(FLORES_URL, partial)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != FLORES_ARCHIVE_SHA256:
        raise ValueError(f"unexpected FLORES archive sha256: {digest}")
    extracted = cache / "flores200_dataset"
    marker = extracted / f".complete-{split}"
    if not marker.exists():
        with tarfile.open(archive) as bundle:
            members = [
                (member, member.name.removeprefix("./"))
                for member in bundle.getmembers()
                if member.isfile()
                and member.name.removeprefix("./").startswith(f"flores200_dataset/{split}/")
                and member.name.endswith(f".{split}")
            ]
            for member, normalized_name in members:
                destination = cache / normalized_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"missing archive member: {member.name}")
                destination.write_bytes(source.read())
        marker.touch()
    return extracted / split


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def draw_ui(width: int, height: int, palette, rng: random.Random) -> Image.Image:
    background, surface, _, accent, border = palette
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    margin = max(12, width // 24)
    radius = max(8, width // 60)
    rounded(draw, (margin, margin, width - margin, height - margin), radius, surface, border)
    header_height = max(30, height // 11)
    draw.line((margin, margin + header_height, width - margin, margin + header_height), fill=border)
    dot = max(3, width // 180)
    for index, color in enumerate(("#ff6258", "#ffbd2e", "#28c840")):
        x = margin + 14 + index * dot * 3
        draw.ellipse((x, margin + 12, x + dot * 2, margin + 12 + dot * 2), fill=color)
    sidebar = width // rng.choice((5, 6, 7))
    draw.rectangle((margin, margin + header_height + 1, margin + sidebar, height - margin), fill=background)
    for row in range(5):
        y = margin + header_height + 20 + row * max(18, height // 14)
        draw.rounded_rectangle((margin + 12, y, margin + sidebar - 12, y + 6), radius=3, fill=border)
    left = margin + sidebar + max(14, width // 32)
    top = margin + header_height + max(14, height // 18)
    right = width - margin - max(14, width // 32)
    bottom = height - margin - max(18, height // 10)
    rounded(draw, (left, top, right, bottom), radius, surface, border)
    button_width = min(max(70, width // 5), right - left - 20)
    button_height = max(22, height // 16)
    rounded(
        draw,
        (right - button_width - 12, bottom - button_height - 12, right - 12, bottom - 12),
        button_height // 3,
        accent,
    )
    if rng.random() < 0.6:
        input_top = bottom - button_height * 2 - 26
        rounded(draw, (left + 12, input_top, right - 12, input_top + button_height), 6, background, border)
    return image


def text_fragment(sentence: str, rng: random.Random) -> str:
    words = sentence.split()
    if len(words) >= 8:
        start = rng.randint(0, max(0, len(words) - 8))
        return " ".join(words[start : start + rng.randint(3, 8)])
    characters = list(sentence)
    width = min(len(characters), rng.randint(6, 24))
    start = rng.randint(0, max(0, len(characters) - width))
    return "".join(characters[start : start + width])


def generate_one(job) -> dict:
    out, split, language, sample_index, sentence, font_family, fontconfig, seed = job
    rng = random.Random(seed)
    width, height = rng.choice(RESOLUTIONS)
    palette = rng.choice(PALETTES)
    image = draw_ui(width, height, palette, rng)
    fragment = text_fragment(sentence, rng)
    language_dir = out / language
    language_dir.mkdir(parents=True, exist_ok=True)
    existing = list(language_dir.glob(f"{sample_index:04d}-*.png"))
    if existing:
        destination = existing[0]
        data = destination.read_bytes()
        with Image.open(io.BytesIO(data)) as saved:
            width, height = saved.size
        digest = hashlib.sha256(data).hexdigest()
        return make_record(
            out, destination, split, language, sample_index, fragment, digest, width, height
        )
    temporary = language_dir / f".{sample_index:04d}.text.png"
    render_text(
        temporary,
        font_family,
        fragment,
        palette[2],
        rng,
        fontconfig,
    )
    with Image.open(temporary) as rendered:
        overlay = rendered.convert("RGBA")
    temporary.unlink()
    x = max(width // 4, int(width * rng.uniform(0.28, 0.38)))
    y = max(height // 5, int(height * rng.uniform(0.22, 0.34)))
    image.paste(overlay, (x, y), overlay)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    destination = language_dir / f"{sample_index:04d}-{digest[:16]}.png"
    destination.write_bytes(data)
    return make_record(
        out, destination, split, language, sample_index, fragment, digest, width, height
    )


def make_record(
    out: Path,
    destination: Path,
    split: str,
    language: str,
    sample_index: int,
    text: str,
    digest: str,
    width: int,
    height: int,
) -> dict:
    return {
        "source": "flores200_balanced_synthetic_ui",
        "record_id": sample_index,
        "sha256": digest,
        "width": width,
        "height": height,
        "language": language,
        "script": language.rsplit("_", 1)[-1],
        "split": split,
        "text": text,
        "license": "CC-BY-SA-4.0 text; Clark Labs generated raster layout",
        "hf_source": FLORES_HF_SOURCE,
        "hf_revision": FLORES_HF_REVISION,
        "upstream_url": FLORES_URL,
        "upstream_archive_sha256": FLORES_ARCHIVE_SHA256,
        "local_path": str(destination.relative_to(out)),
    }


def main() -> None:
    args = parse_args()
    if args.reserve_per_language < 0:
        raise ValueError("reserve-per-language must be non-negative")
    if bool(args.exclude_text_split) != bool(args.exclude_text_per_language):
        raise ValueError(
            "exclude-text-split and exclude-text-per-language must be used together"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    excluded_hashes = load_excluded_hashes(args.exclude_manifest)
    font_assets = prepare_fontconfig(args.cache.parent / "fonts")
    source_root = fetch_flores(args.cache, args.split)
    all_language_files = sorted(source_root.glob(f"*.{args.split}"))
    if len(all_language_files) != 204:
        raise ValueError(
            "expected 204 FLORES language-script variants, "
            f"found {len(all_language_files)}"
        )
    indexed_language_files = list(enumerate(all_language_files))
    if args.language:
        requested = set(args.language)
        indexed_language_files = [
            (index, path)
            for index, path in indexed_language_files
            if path.stem in requested
        ]
        found = {path.stem for _, path in indexed_language_files}
        if found != requested:
            raise ValueError(
                "unknown FLORES language-script variants: "
                + ", ".join(sorted(requested - found))
            )
    excluded_text_root = (
        fetch_flores(args.cache, args.exclude_text_split)
        if args.exclude_text_split
        else None
    )
    jobs = []
    excluded_sentences = 0
    for language_index, source in indexed_language_files:
        language = source.stem
        script = language.split("_")[-1]
        font_family = SCRIPT_FONTS.get(script, "Arial Unicode MS")
        sentences = [
            sanitized
            for line in source.read_text().splitlines()
            if (sanitized := sanitize_text(line))
        ]
        if excluded_text_root is not None:
            excluded_source = excluded_text_root / (
                f"{language}.{args.exclude_text_split}"
            )
            training_sentences = [
                sanitized
                for line in excluded_source.read_text().splitlines()
                if (sanitized := sanitize_text(line))
            ]
            exclusion_rng = random.Random(args.exclude_text_seed + language_index)
            exclusion_rng.shuffle(training_sentences)
            excluded = set(
                training_sentences[: args.exclude_text_per_language]
            )
            before = len(sentences)
            sentences = [
                sentence for sentence in sentences
                if sentence not in excluded
            ]
            excluded_sentences += before - len(sentences)
        candidate_count = args.per_language + args.reserve_per_language
        if len(sentences) < candidate_count:
            raise ValueError(f"{language} only has {len(sentences)} sentences")
        rng = random.Random(args.seed + language_index)
        rng.shuffle(sentences)
        for sample_index, sentence in enumerate(sentences[:candidate_count]):
            jobs.append(
                (
                    args.out,
                    args.split,
                    language,
                    sample_index,
                    sentence,
                    font_family,
                    font_assets["config"],
                    args.seed + language_index * 100_000 + sample_index,
                )
            )
    manifest = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(generate_one, jobs), start=1):
            manifest.append(record)
            if index % 1000 == 0:
                print(json.dumps({"generated": index, "target": len(jobs)}), flush=True)
    if args.exclude_manifest or args.reserve_per_language:
        manifest = select_disjoint_records(
            manifest,
            args.per_language,
            excluded_hashes,
        )
    with (args.out / "manifest.jsonl").open("w") as handle:
        for record in manifest:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "images": len(manifest),
                "candidate_images": len(jobs),
                "languages": len(indexed_language_files),
                "images_per_language": args.per_language,
                "reserve_per_language": args.reserve_per_language,
                "excluded_manifests": [str(path) for path in args.exclude_manifest],
                "excluded_hashes": len(excluded_hashes),
                "excluded_text_split": args.exclude_text_split,
                "excluded_text_per_language": args.exclude_text_per_language,
                "excluded_text_seed": args.exclude_text_seed,
                "excluded_sentences": excluded_sentences,
                "discarded_candidates": len(jobs) - len(manifest),
                "split": args.split,
                "resolutions": RESOLUTIONS,
                "seed": args.seed,
                "source": FLORES_HF_SOURCE,
                "source_revision": FLORES_HF_REVISION,
                "license": "CC-BY-SA-4.0",
                "text_renderer": "pango-view",
                "reject_unknown_glyphs": True,
                "font_assets": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in font_assets.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
