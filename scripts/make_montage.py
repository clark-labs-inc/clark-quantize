"""Build a labelled comparison grid from generated images.

Usage:
  python scripts/make_montage.py --out assets/compare.png \
      fp16:"FP16 baseline" opt_g128:"optimal g128" opt_g32:"optimal g32"
Each arg is TAG:LABEL; expects out/<TAG>_<i>.png for i in 0..n-1.
"""
import argparse
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cols", nargs="+", help="TAG:LABEL entries")
    ap.add_argument("--indir", default="out")
    ap.add_argument("--out", default="assets/compare.png")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cell", type=int, default=460)
    a = ap.parse_args()

    cols = [c.split(":", 1) for c in a.cols]
    pad, head = 30, 40
    W = len(cols) * a.cell + (len(cols) + 1) * pad
    H = a.n * a.cell + (a.n + 1) * pad + head
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    for c, (tag, label) in enumerate(cols):
        x = pad + c * (a.cell + pad)
        d.text((x + 4, 10), label, fill="black", font=f)
        for r in range(a.n):
            y = head + pad + r * (a.cell + pad)
            img = Image.open(f"{a.indir}/{tag}_{r}.png").resize((a.cell, a.cell))
            canvas.paste(img, (x, y))
    canvas.save(a.out)
    print("wrote", a.out, canvas.size)


if __name__ == "__main__":
    main()
