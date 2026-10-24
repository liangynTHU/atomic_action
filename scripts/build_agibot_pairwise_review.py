#!/usr/bin/env python3
"""Build a small, disposable visual atlas for pairwise AgiBot label review."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CASES = {
    "coffee": ("task_431/episode_000047", "agibot_glm53_grounded_v7_object_grounding_trial_20261024"),
    "iron": ("task_440/episode_000447", "agibot_glm53_grounded_v7_object_grounding_trial_20261024"),
    "bin": ("task_521/episode_000216", "agibot_glm53_grounded_v7_object_grounding_trial_v2_20261024"),
}
CAMERAS = ("head", "hand_left", "hand_right")
TILE = (400, 300)
HEADER = 44


def load_case(root, case, *, trial_root=None, reference="baseline", cached=False):
    episode_id, trial_name = CASES[case]
    run = root / "artifacts/agibot_glm53_grounded_v7_random10_20261024_b_run"
    accepted = next(json.loads(line) for line in (run / "accepted.jsonl").open()
                    if json.loads(line)["episode_id"] == episode_id)
    source = next(json.loads(line) for line in (run / "inventory/episodes.jsonl").open()
                  if json.loads(line)["episode_id"] == episode_id)
    old = json.loads((Path(accepted["path"]) / "vla.json").read_text())["segments"]
    previous_trial = root / "artifacts" / trial_name / episode_id.replace("/", "_")
    if reference == "previous":
        old = json.loads((previous_trial / "segments.json").read_text())
    trial = (trial_root or root / "artifacts" / trial_name) / episode_id.replace("/", "_")
    if (trial / "segments.json").exists():
        new = json.loads((trial / "segments.json").read_text())
    elif cached:
        batch_size = json.loads((trial / "INPUTS.json").read_text())["batch_size"]
        new = []
        for batch_index in range((len(old) + batch_size - 1) // batch_size):
            path = trial / "labels" / f"batch_{batch_index:05d}.json"
            if not path.exists():
                break
            labels = json.loads(path.read_text())["labels"]["segments"]
            if len(labels) != min(batch_size, len(old) - len(new)):
                raise ValueError(f"incomplete accepted batch: {path}")
            for label in labels:
                segment = dict(old[label["segment_index"]])
                segment["sub_task"] = label["sub_task"]
                new.append(segment)
    else:
        raise FileNotFoundError(trial / "segments.json")
    media = run / "episodes" / source["key"][:2] / source["key"] / "media"
    if not cached and len(old) != len(new):
        raise ValueError("segment counts differ")
    for before, after in zip(old, new):
        if any(before[key] != after[key] for key in
               ("segment_index", "start_frame", "end_frame", "left_action", "right_action")):
            raise ValueError("segment boundaries or primitive actions differ")
    return old, new, media


def draw_segment(atlas, row, segment, media):
    draw = ImageDraw.Draw(atlas)
    frames = np.linspace(segment["start_frame"], segment["end_frame"] + 1, 5, dtype=int)
    top = row * (HEADER + len(CAMERAS) * TILE[1])
    draw.rectangle((0, top, atlas.width, top + HEADER), fill="#102637")
    draw.text((12, top + 12),
              f"SEGMENT {segment['segment_index'] + 1} | actions {segment['start_frame']}..{segment['end_frame']} | observations {list(frames)}",
              fill="white")
    for camera_index, camera in enumerate(CAMERAS):
        for column, frame in enumerate(frames):
            path = media / camera / f"{frame:07d}.jpg"
            with Image.open(path) as original:
                tile = original.convert("RGB").resize(TILE, Image.Resampling.LANCZOS)
            left = column * TILE[0]
            y = top + HEADER + camera_index * TILE[1]
            atlas.paste(tile, (left, y))
            draw.rectangle((left, y, left + 137, y + 21), fill="#102637")
            draw.text((left + 5, y + 4), f"{camera} f{frame}", fill="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("start", type=int, help="1-based display segment")
    parser.add_argument("--count", type=int, default=2, choices=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-root", type=Path, help="new trial output root; defaults to the original trial")
    parser.add_argument("--reference", choices=("baseline", "previous"), default="baseline")
    parser.add_argument("--cached", action="store_true", help="read accepted batches before segments.json exists")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    old, new, media = load_case(root, args.case, trial_root=args.trial_root,
                                reference=args.reference, cached=args.cached)
    if not 1 <= args.start or args.start + args.count - 1 > len(new):
        parser.error(f"segment range outside available trial output ({len(new)} segments)")
    atlas = Image.new("RGB", (5 * TILE[0], args.count * (HEADER + 3 * TILE[1])))
    for row, index in enumerate(range(args.start - 1, args.start - 1 + args.count)):
        draw_segment(atlas, row, old[index], media)
        print(json.dumps({"case": args.case, "segment": index + 1,
                          "start_frame": old[index]["start_frame"],
                          "target_frame": old[index]["end_frame"] + 1,
                          "old": old[index]["sub_task"], "new": new[index]["sub_task"]},
                         ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(args.output, quality=82, optimize=True)


if __name__ == "__main__":
    main()
