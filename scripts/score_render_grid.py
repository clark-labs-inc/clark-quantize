"""Score FP16-vs-candidate render grids.

The Sana training renders are contact sheets with FP16 references on the top row
and the candidate model on the bottom row. This script gives a cheap, deterministic
visual reward proxy for ranking checkpoints before training any learned critic.

It is intentionally not an OCR/object detector replacement. It is a gate: if this
kind of visible-image reward cannot reproduce known visual preferences, we should
not use a learned value target derived from it for QAT gradients.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


EPS = 1e-8


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_gray(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def sobel(gray: np.ndarray) -> np.ndarray:
    gx, gy = sobel_xy(gray)
    return np.sqrt(gx * gx + gy * gy)


def sobel_xy(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2 * p[1:-1, :-2]
        + 2 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    return gx, gy


def lowpass(x: np.ndarray, radius: float = 5.0) -> np.ndarray:
    img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def split_grid(img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w, _ = img.shape
    if h % 2 != 0:
        raise ValueError(f"expected even image height for two-row grid, got {h}")
    cell = h // 2
    cols = max(1, w // cell)
    if cols * cell != w:
        raise ValueError(f"expected width divisible by cell size {cell}, got width {w}")
    pairs = []
    for i in range(cols):
        x0 = i * cell
        x1 = x0 + cell
        pairs.append((img[:cell, x0:x1], img[cell:, x0:x1]))
    return pairs


def exp_score(x: float, scale: float) -> float:
    return float(math.exp(-max(0.0, x) / max(scale, EPS)))


def ratio_score(a: float, b: float) -> float:
    return float(math.exp(-abs(math.log(max(a, EPS) / max(b, EPS)))))


def overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def connected_components(mask: np.ndarray, min_area: int = 8) -> list[dict[str, float]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        xs = np.nonzero(mask[y] & ~seen[y])[0]
        for x in xs:
            if seen[y, x]:
                continue
            stack = [(int(x), int(y))]
            seen[y, x] = True
            pts = []
            while stack:
                px, py = stack.pop()
                pts.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(pts) >= min_area:
                arr = np.asarray(pts, dtype=np.float32)
                x0, y0 = arr.min(axis=0)
                x1, y1 = arr.max(axis=0) + 1
                comps.append(
                    {
                        "area": float(len(pts)),
                        "x": float(arr[:, 0].mean()),
                        "y": float(arr[:, 1].mean()),
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                    }
                )
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def roi(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(round(box[0] * w))
    y0 = int(round(box[1] * h))
    x1 = int(round(box[2] * w))
    y1 = int(round(box[3] * h))
    return img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]


def saturation(x: np.ndarray) -> np.ndarray:
    return x.max(axis=-1) - x.min(axis=-1)


def foreground_mask(x: np.ndarray) -> np.ndarray:
    gray = to_gray(x)
    sat = saturation(x)
    edge = sobel(gray)
    return (sat > 0.12) | (gray < 0.42) | (edge > np.quantile(edge, 0.82))


def mask_area_score(ref_mask: np.ndarray, cand_mask: np.ndarray) -> float:
    return ratio_score(float(cand_mask.mean()), float(ref_mask.mean()))


def dark_structure_score(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    """Proxy for text/glyph structure in the first hard prompt.

    We look in the upper-left wall region, where the hard text prompt appears in
    the current Sana render suite. This is deliberately narrow so it does not
    pretend to solve general OCR.
    """
    h, w, _ = ref.shape
    y1 = int(h * 0.48)
    x1 = int(w * 0.62)
    rg = to_gray(ref[:y1, :x1])
    cg = to_gray(cand[:y1, :x1])
    ref_mask = rg < min(0.32, float(np.quantile(rg, 0.12)))
    cand_mask_full = cg < min(0.32, float(np.quantile(cg, 0.12)))
    box = bbox_from_mask(ref_mask, pad=max(8, x1 // 60))
    if box is None:
        cand_mask = cand_mask_full
        ref_focus = ref_mask
    else:
        bx0, by0, bx1, by1 = box
        cand_mask = cand_mask_full[by0:by1, bx0:bx1]
        ref_focus = ref_mask[by0:by1, bx0:bx1]
    union = np.logical_or(ref_focus, cand_mask).sum()
    inter = np.logical_and(ref_focus, cand_mask).sum()
    ref_area = float(ref_focus.mean())
    cand_area = float(cand_mask.mean())
    iou = float(inter / max(union, 1))
    area_ratio = cand_area / max(ref_area, EPS)
    area_score = ratio_score(cand_area, ref_area)
    ref_comps = connected_components(ref_focus, min_area=max(4, ref_focus.size // 2000))
    cand_comps = connected_components(cand_mask, min_area=max(4, cand_mask.size // 2000))
    comp_score = ratio_score(float(len(cand_comps) + 1), float(len(ref_comps) + 1))
    edge_ref = sobel(rg if box is None else rg[by0:by1, bx0:bx1])
    edge_cand = sobel(cg if box is None else cg[by0:by1, bx0:bx1])
    edge_score = exp_score(float(np.abs(edge_ref - edge_cand).mean() / max(edge_ref.mean(), EPS)), 0.75)
    stray = 0.0
    if box is not None:
        outside = cand_mask_full.copy()
        outside[by0:by1, bx0:bx1] = False
        stray = float(outside.mean() / max(cand_mask_full.mean(), EPS))
    stray_score = exp_score(stray, 0.35)
    text_proxy = 0.35 * iou + 0.20 * area_score + 0.20 * edge_score + 0.15 * comp_score + 0.10 * stray_score
    return {
        "dark_iou": iou,
        "dark_area_ratio": float(area_ratio),
        "dark_area_score": float(area_score),
        "glyph_component_score": float(comp_score),
        "text_edge_score": float(edge_score),
        "text_stray_score": float(stray_score),
        "text_proxy": float(text_proxy),
    }


def wine_scene_score(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    # The wine prompt is panel 1 in the hard render suite.
    rg = roi(ref, (0.22, 0.05, 0.88, 0.95))
    cg = roi(cand, (0.22, 0.05, 0.88, 0.95))
    ref_red = (rg[..., 0] > rg[..., 1] * 1.08) & (rg[..., 0] > rg[..., 2] * 1.08) & (to_gray(rg) < 0.65)
    cand_red = (cg[..., 0] > cg[..., 1] * 1.08) & (cg[..., 0] > cg[..., 2] * 1.08) & (to_gray(cg) < 0.65)
    bottle_score = 0.5 * overlap_score(ref_red, cand_red) + 0.5 * mask_area_score(ref_red, cand_red)

    # Weak "separate object" proxy: bright/edgy, low-saturation components outside the main red bottle.
    rg_gray, cg_gray = to_gray(rg), to_gray(cg)
    re, ce = sobel(rg_gray), sobel(cg_gray)
    ref_glass = (rg_gray > 0.42) & (saturation(rg) < 0.20) & (re > np.quantile(re, 0.84)) & ~ref_red
    cand_glass = (cg_gray > 0.42) & (saturation(cg) < 0.20) & (ce > np.quantile(ce, 0.84)) & ~cand_red
    glass_score = 0.5 * overlap_score(ref_glass, cand_glass) + 0.5 * mask_area_score(ref_glass, cand_glass)

    ref_grapes = roi(ref, (0.42, 0.65, 0.68, 0.96))
    cand_grapes = roi(cand, (0.42, 0.65, 0.68, 0.96))
    grape_score = exp_score(float(np.abs(sobel(to_gray(ref_grapes)) - sobel(to_gray(cand_grapes))).mean()), 0.11)
    object_score = 0.45 * bottle_score + 0.35 * glass_score + 0.20 * grape_score
    return {
        "bottle_present": float(bottle_score),
        "glass_present_proxy": float(glass_score),
        "grapes_present_proxy": float(grape_score),
        "objects_proxy": float(object_score),
    }


def harp_score(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    rr = roi(ref, (0.16, 0.02, 0.88, 0.98))
    cr = roi(cand, (0.16, 0.02, 0.88, 0.98))
    rg, cg = to_gray(rr), to_gray(cr)
    rgx, rgy = sobel_xy(rg)
    cgx, cgy = sobel_xy(cg)
    ref_vertical = np.abs(rgx) > np.quantile(np.abs(rgx), 0.88)
    cand_vertical = np.abs(cgx) > np.quantile(np.abs(cgx), 0.88)
    string_overlap = overlap_score(ref_vertical, cand_vertical)
    string_density = mask_area_score(ref_vertical, cand_vertical)
    frame_ref = (rr[..., 0] > rr[..., 2] * 1.15) & (rr[..., 0] > rr[..., 1] * 0.75) & (saturation(rr) > 0.12)
    frame_cand = (cr[..., 0] > cr[..., 2] * 1.15) & (cr[..., 0] > cr[..., 1] * 0.75) & (saturation(cr) > 0.12)
    frame_score = 0.5 * overlap_score(frame_ref, frame_cand) + 0.5 * mask_area_score(frame_ref, frame_cand)
    # Encourage vertical structure more than arbitrary edge magnitude.
    vertical_energy_score = ratio_score(float(np.abs(cgx).mean()), float(np.abs(rgx).mean()))
    horizontal_penalty = ratio_score(float(np.abs(cgy).mean()), float(np.abs(rgy).mean()))
    geometry = 0.35 * string_overlap + 0.20 * string_density + 0.25 * frame_score + 0.10 * vertical_energy_score + 0.10 * horizontal_penalty
    return {
        "harp_string_overlap": float(string_overlap),
        "harp_string_density": float(string_density),
        "harp_frame": float(frame_score),
        "harp_geometry": float(geometry),
    }


def horse_person_score(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    horse_ref = foreground_mask(roi(ref, (0.12, 0.08, 0.62, 0.95)))
    horse_cand = foreground_mask(roi(cand, (0.12, 0.08, 0.62, 0.95)))
    person_ref = foreground_mask(roi(ref, (0.55, 0.08, 0.96, 0.95)))
    person_cand = foreground_mask(roi(cand, (0.55, 0.08, 0.96, 0.95)))
    horse_present = 0.45 * overlap_score(horse_ref, horse_cand) + 0.55 * mask_area_score(horse_ref, horse_cand)
    person_present = 0.45 * overlap_score(person_ref, person_cand) + 0.55 * mask_area_score(person_ref, person_cand)
    whole_ref = foreground_mask(ref)
    whole_cand = foreground_mask(cand)
    ref_comps = connected_components(whole_ref, min_area=max(64, whole_ref.size // 1200))[:4]
    cand_comps = connected_components(whole_cand, min_area=max(64, whole_cand.size // 1200))[:4]
    count_score = ratio_score(float(len(cand_comps) + 1), float(len(ref_comps) + 1))
    relation_score = 0.40 * horse_present + 0.35 * person_present + 0.25 * count_score
    return {
        "horse_present_proxy": float(horse_present),
        "person_present_proxy": float(person_present),
        "horse_person_count_proxy": float(count_score),
        "relation_proxy": float(relation_score),
    }


def score_pair(ref: np.ndarray, cand: np.ndarray, panel_index: int) -> dict[str, float]:
    diff = np.abs(ref - cand)
    mae = float(diff.mean())
    mse = float(((ref - cand) ** 2).mean())
    psnr = float(-10.0 * math.log10(max(mse, EPS)))

    ref_low = lowpass(ref)
    cand_low = lowpass(cand)
    low_mae = float(np.abs(ref_low - cand_low).mean())

    re = sobel(to_gray(ref))
    ce = sobel(to_gray(cand))
    edge_mae = float(np.abs(re - ce).mean())
    edge_ref = float(re.mean())
    edge_cand = float(ce.mean())
    edge_ratio = edge_cand / max(edge_ref, EPS)

    teacher_sim = exp_score(mae, 0.18)
    low_sim = exp_score(low_mae, 0.14)
    edge_sim = exp_score(edge_mae / max(edge_ref, EPS), 0.9)
    detail_balance = math.exp(-abs(math.log(max(edge_ratio, EPS))))

    text = {
        "text_proxy": 0.0,
        "dark_iou": 0.0,
        "dark_area_ratio": 0.0,
        "dark_area_score": 0.0,
        "glyph_component_score": 0.0,
        "text_edge_score": 0.0,
        "text_stray_score": 0.0,
    }
    objects = {"objects_proxy": 0.0, "bottle_present": 0.0, "glass_present_proxy": 0.0, "grapes_present_proxy": 0.0}
    harp = {"harp_geometry": 0.0, "harp_string_overlap": 0.0, "harp_string_density": 0.0, "harp_frame": 0.0}
    relation = {"relation_proxy": 0.0, "horse_present_proxy": 0.0, "person_present_proxy": 0.0, "horse_person_count_proxy": 0.0}
    if panel_index == 0:
        text = dark_structure_score(ref, cand)
    elif panel_index == 1:
        objects = wine_scene_score(ref, cand)
    elif panel_index == 2:
        harp = harp_score(ref, cand)
    elif panel_index == 3:
        relation = horse_person_score(ref, cand)

    heads = {
        "teacher": float(teacher_sim),
        "text": float(text["text_proxy"]) if panel_index == 0 else float(0.55 * teacher_sim + 0.45 * edge_sim),
        "objects": float(objects["objects_proxy"]) if panel_index == 1 else float(0.55 * teacher_sim + 0.45 * low_sim),
        "relations": float(relation["relation_proxy"]) if panel_index == 3 else float(0.50 * teacher_sim + 0.50 * low_sim),
        "geometry": float(harp["harp_geometry"]) if panel_index == 2 else float(0.55 * edge_sim + 0.45 * detail_balance),
        "composition": float(0.55 * low_sim + 0.30 * teacher_sim + 0.15 * detail_balance),
    }

    reward = (
        0.26 * heads["teacher"]
        + 0.16 * heads["text"]
        + 0.17 * heads["objects"]
        + 0.17 * heads["relations"]
        + 0.14 * heads["geometry"]
        + 0.10 * heads["composition"]
    )
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "low_mae": low_mae,
        "edge_mae": edge_mae,
        "edge_ref": edge_ref,
        "edge_cand": edge_cand,
        "edge_ratio": float(edge_ratio),
        "teacher_sim": float(teacher_sim),
        "low_sim": float(low_sim),
        "edge_sim": float(edge_sim),
        "detail_balance": float(detail_balance),
        "reward": float(reward),
        "heads": heads,
        **text,
        **objects,
        **harp,
        **relation,
    }


def score_grid(path: str) -> dict:
    img = load_rgb(path)
    pairs = split_grid(img)
    panels = [score_pair(ref, cand, i) for i, (ref, cand) in enumerate(pairs)]
    keys = ("reward", "teacher_sim", "edge_sim", "low_sim", "detail_balance", "text_proxy", "objects_proxy", "relation_proxy", "harp_geometry", "psnr")
    mean = {k: float(np.mean([p[k] for p in panels])) for k in keys}
    hard_min = {k + "_min": float(np.min([p[k] for p in panels])) for k in keys if k != "psnr"}
    head_names = ("teacher", "text", "objects", "relations", "geometry", "composition")
    head_mean = {k: float(np.mean([p["heads"][k] for p in panels])) for k in head_names}
    head_min = {k + "_min": float(np.min([p["heads"][k] for p in panels])) for k in head_names}
    score = 100.0 * (0.82 * mean["reward"] + 0.18 * hard_min["reward_min"])
    return {
        "path": path,
        "label": Path(path).stem,
        "panels": panels,
        "mean": mean,
        "hard_min": hard_min,
        "head_mean": head_mean,
        "head_min": head_min,
        "score": float(score),
        "cols": len(pairs),
    }


def write_markdown(rows: list[dict], path: str) -> None:
    lines = [
        "| rank | score | label | teacher | text | objects | relations | geometry | composition | path |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(rows, 1):
        h = row["head_mean"]
        lines.append(
            "| {rank} | {score:.2f} | {label} | {teacher:.3f} | {text:.3f} | "
            "{objects:.3f} | {relations:.3f} | {geometry:.3f} | {composition:.3f} | `{path}` |".format(
                rank=i,
                score=row["score"],
                label=row["label"],
                teacher=h["teacher"],
                text=h["text"],
                objects=h["objects"],
                relations=h["relations"],
                geometry=h["geometry"],
                composition=h["composition"],
                path=row["path"],
            )
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("renders", nargs="+", help="2-row FP16/candidate render grids")
    ap.add_argument("--out-json", default="render_reward_scores.json")
    ap.add_argument("--out-md", default="render_reward_scores.md")
    args = ap.parse_args()

    rows = []
    for path in args.renders:
        try:
            rows.append(score_grid(path))
        except Exception as exc:
            rows.append({"path": path, "label": Path(path).stem, "error": str(exc), "score": -1.0})
    rows.sort(key=lambda r: r.get("score", -1.0), reverse=True)

    payload = {
        "description": "Vector visual reward proxy for two-row FP16-vs-candidate render grids.",
        "warning": "Use this as a rank/selection gate. Do not use as a training reward until pairwise validation passes.",
        "heads": ["teacher", "text", "objects", "relations", "geometry", "composition"],
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown([r for r in rows if "error" not in r], args.out_md)
    for i, row in enumerate(rows, 1):
        if "error" in row:
            print(f"{i:02d} score=ERR label={row['label']} error={row['error']}")
        else:
            print(f"{i:02d} score={row['score']:.2f} label={row['label']} path={row['path']}")


if __name__ == "__main__":
    main()
