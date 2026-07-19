"""Build panel-level VLM pairwise judge jobs from render reward preferences.

The render grids are two-row contact sheets: FP16 references on the top row and
candidate renders on the bottom row. VLM judges should see one full-resolution
panel at a time, not the whole montage, so this script crops reference/A/B panels
and emits JSONL jobs for offline labeling.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


GOAL_TO_RUBRICS = {
    "text": ["wall_text"],
    "objects": ["wine_glass"],
    "relations": ["horse_person"],
    "geometry": ["harp_strings"],
    "composition": ["wall_text", "wine_glass", "harp_strings", "horse_person"],
    "overall": ["wall_text", "wine_glass", "harp_strings", "horse_person"],
    "score": ["wall_text", "wine_glass", "harp_strings", "horse_person"],
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180]


def split_grid(path: str | Path) -> list[tuple[Image.Image, Image.Image]]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h % 2 != 0:
        raise ValueError(f"expected even height for two-row render grid: {path}")
    cell = h // 2
    if w % cell != 0:
        raise ValueError(f"expected width divisible by panel size {cell}: {path}")
    panels = []
    for i in range(w // cell):
        x0 = i * cell
        panels.append(
            (
                img.crop((x0, 0, x0 + cell, cell)),
                img.crop((x0, cell, x0 + cell, h)),
            )
        )
    return panels


def crop_panel(
    *,
    render_path: str | Path,
    render_label: str,
    panel_index: int,
    kind: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug(render_label)}_p{panel_index}_{kind}.png"
    if out.exists():
        return out
    panels = split_grid(render_path)
    if panel_index >= len(panels):
        raise IndexError(f"panel {panel_index} missing in {render_path}; found {len(panels)} panels")
    ref, cand = panels[panel_index]
    (ref if kind == "ref" else cand).save(out)
    return out


def row_index(scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [r for r in scores.get("rows", []) if "error" not in r]
    return {r["label"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSON from score_render_grid.py")
    ap.add_argument("--preferences", required=True, help="Known pairwise preference JSON")
    ap.add_argument("--rubrics", required=True, help="Rubric JSON, usually data/sana_hard_prompt_rubrics.json")
    ap.add_argument("--panel-dir", required=True, help="Directory for cropped panel PNGs")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--include-goal", action="append", default=[], help="Optional goal filter; repeatable")
    args = ap.parse_args()

    scores = load_json(args.scores)
    prefs = load_json(args.preferences).get("preferences", [])
    rubrics = load_json(args.rubrics).get("rubrics", {})
    rows = row_index(scores)
    panel_dir = Path(args.panel_dir)
    include_goals = set(args.include_goal)

    jobs = []
    for pref_index, pref in enumerate(prefs):
        goal = pref["goal"]
        if include_goals and goal not in include_goals:
            continue
        better = pref["better"]
        worse = pref["worse"]
        if better not in rows or worse not in rows:
            continue
        rubric_ids = GOAL_TO_RUBRICS.get(goal, GOAL_TO_RUBRICS["overall"])
        for rubric_id in rubric_ids:
            rubric = rubrics[rubric_id]
            panel_index = int(rubric["panel_index"])
            better_row = rows[better]
            worse_row = rows[worse]
            ref_path = crop_panel(
                render_path=better_row["path"],
                render_label=better,
                panel_index=panel_index,
                kind="ref",
                out_dir=panel_dir,
            )
            cand_a = crop_panel(
                render_path=better_row["path"],
                render_label=better,
                panel_index=panel_index,
                kind="candidate",
                out_dir=panel_dir,
            )
            cand_b = crop_panel(
                render_path=worse_row["path"],
                render_label=worse,
                panel_index=panel_index,
                kind="candidate",
                out_dir=panel_dir,
            )
            job_id = f"pref{pref_index:03d}_{goal}_{rubric_id}_{slug(better)}__vs__{slug(worse)}"
            jobs.append(
                {
                    "id": job_id,
                    "goal": goal,
                    "prompt_id": rubric["prompt_id"],
                    "prompt": rubric["prompt"],
                    "rubric": rubric,
                    "reference": str(ref_path),
                    "candidate_a": str(cand_a),
                    "candidate_b": str(cand_b),
                    "candidate_a_label": better,
                    "candidate_b_label": worse,
                    "expected_winner": "A",
                    "source_preference": pref,
                }
            )

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")
    print(f"wrote {len(jobs)} jobs to {out}")
    print(f"panel_dir={panel_dir}")


if __name__ == "__main__":
    main()
