"""Pinned, glyph-complete Pango rendering for multilingual UI canvases."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image, ImageColor, ImageOps


GOOGLE_FONTS_REVISION = "7ff85c87f93ea6cca5f41c69f2e4edcb90240f26"
TIFINAGH_FILENAME = "NotoSansTifinagh-Regular.ttf"
TIFINAGH_SHA256 = "b08755bdee9835ebbca1095460e36a9e2f3ae396919fd0fa2377d9b4a3388f7e"
TIFINAGH_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    f"{GOOGLE_FONTS_REVISION}/ofl/notosanstifinagh/{TIFINAGH_FILENAME}"
)

SCRIPT_FONTS = {
    "Arab": ".SF Arabic",
    "Armn": "Noto Sans Armenian",
    "Beng": "Bangla Sangam MN",
    "Deva": "Devanagari Sangam MN",
    "Ethi": "Kefa III",
    "Geor": ".SF Georgian",
    "Gujr": "Gujarati Sangam MN",
    "Guru": "Gurmukhi Sangam MN",
    "Hang": "Apple SD Gothic Neo",
    "Hans": "Hiragino Sans GB",
    "Hant": "Arial Unicode MS",
    "Hebr": ".SF Hebrew",
    "Jpan": "Hiragino Sans",
    "Khmr": "Khmer Sangam MN",
    "Knda": "Kannada Sangam MN",
    "Laoo": "Lao Sangam MN",
    "Mlym": "Malayalam Sangam MN",
    "Mymr": "Noto Sans Myanmar",
    "Olck": "Noto Sans Ol Chiki",
    "Orya": "Noto Sans Oriya",
    "Sinh": "Sinhala Sangam MN",
    "Taml": "Tamil Sangam MN",
    "Telu": "Telugu Sangam MN",
    "Tfng": "Noto Sans Tifinagh",
    "Thai": "Thonburi",
    "Tibt": "Kailasa",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_fontconfig() -> Path:
    candidates = (
        Path("/opt/homebrew/etc/fonts/fonts.conf"),
        Path("/usr/local/etc/fonts/fonts.conf"),
        Path("/etc/fonts/fonts.conf"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("system fontconfig configuration was not found")


def prepare_fontconfig(cache: Path) -> dict:
    directory = cache / f"google-fonts-{GOOGLE_FONTS_REVISION}"
    directory.mkdir(parents=True, exist_ok=True)
    font = directory / TIFINAGH_FILENAME
    if not font.exists():
        partial = font.with_suffix(".partial")
        urllib.request.urlretrieve(TIFINAGH_URL, partial)
        partial.replace(font)
    digest = file_sha256(font)
    if digest != TIFINAGH_SHA256:
        raise ValueError(f"unexpected Tifinagh font sha256: {digest}")
    config = directory / "fonts.conf"
    config.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
        "<fontconfig>\n"
        f'  <include ignore_missing="no">{system_fontconfig()}</include>\n'
        f"  <dir>{directory}</dir>\n"
        "</fontconfig>\n"
    )
    return {
        "config": config,
        "font": font,
        "font_sha256": digest,
        "source_revision": GOOGLE_FONTS_REVISION,
        "source_url": TIFINAGH_URL,
    }


def colorize_mask(mask_path: Path, output: Path, foreground: str) -> None:
    with Image.open(mask_path) as source:
        alpha = ImageOps.invert(source.convert("L"))
    overlay = Image.new("RGBA", alpha.size, (*ImageColor.getrgb(foreground), 255))
    overlay.putalpha(alpha)
    overlay.save(output, format="PNG")


def render_text(
    output: Path,
    font_family: str,
    fragment: str,
    foreground: str,
    rng: random.Random,
    fontconfig: Path,
) -> None:
    mask = output.with_suffix(".pgm")
    layout = output.with_suffix(".layout.json")
    command = [
        "pango-view",
        "--backend=ft2",
        "--no-display",
        "--background=white",
        "--foreground=black",
        "--margin=0",
        "--pixels",
        "--single-par",
        f"--font={font_family} {rng.randint(8, 30)}",
        f"--text={fragment}",
        f"--output={mask}",
        f"--serialize-to={layout}",
    ]
    environment = {**os.environ, "FONTCONFIG_FILE": str(fontconfig)}
    subprocess.run(
        command,
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    metadata = json.loads(layout.read_text())
    unknown = int(metadata["output"]["unknown-glyphs"])
    if unknown:
        mask.unlink(missing_ok=True)
        layout.unlink(missing_ok=True)
        raise ValueError(f"Pango could not render {unknown} glyphs in {fragment!r}")
    colorize_mask(mask, output, foreground)
    mask.unlink()
    layout.unlink()
