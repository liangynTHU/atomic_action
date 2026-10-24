#!/usr/bin/env python3
"""Prepare read-only visual evidence for auditing every generated AgiBot label."""

import argparse
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.agibot.annotate import validate_visual_review, visual_boundary
from atomic_seg.pretrain.agibot.config import Settings
from atomic_seg.pretrain.agibot.runner import episode_dir, episode_from, load_numeric
from atomic_seg.pretrain.agibot.source import iter_inventory
from atomic_seg.pretrain.realomni.storage import atomic_json, read_json, file_hash


def arm_metrics(states, start, target, offset):
    arm = states[start:target + 1, offset:offset + 8]
    return {
        "start_openness": round(float(arm[0, 7]), 3),
        "target_openness": round(float(arm[-1, 7]), 3),
        "openness_range": [round(float(arm[:, 7].min()), 3),
                           round(float(arm[:, 7].max()), 3)],
        "position_path_m": round(float(np.linalg.norm(np.diff(arm[:, :3], axis=0), axis=1).sum()), 4),
        "position_delta_m": [round(float(x), 4) for x in arm[-1, :3] - arm[0, :3]],
    }


def build(run, output, rows_per_sheet=3, include_accepted_batches=False):
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("audit output must not overlap the source run")
    config = read_json(run / "run_config.json")
    manifest = run / "accepted.jsonl"
    accepted = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    by_key = {Path(entry["path"]).name: entry for entry in accepted}
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 22)
    title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 23)
    rows = []
    for source in iter_inventory(run):
        entry = by_key.get(source["key"])
        directory = episode_dir(run, source)
        if entry is not None:
            export = Path(entry["path"])
            segments = read_json(export / "vla.json")["segments"]
            with np.load(export / "reference.npz", allow_pickle=False) as arrays:
                states = arrays["measured_states"].copy()
        elif include_accepted_batches and (directory / "segmented/COMMIT.json").exists():
            fixed = read_json(directory / "segmented/segmentation.json")["segments"]
            segments, previous_boundary = [], None
            batch_size = config["settings"]["batch_size"]
            for batch_index, start in enumerate(range(0, len(fixed), batch_size)):
                cache = directory / "labels" / f"batch_{batch_index:05d}.json"
                if not cache.exists():
                    break
                value = read_json(cache)
                requested = fixed[start:start + batch_size]
                validate_visual_review({**value["labels"], "visual_checks": value["visual_checks"]},
                                       requested, previous_boundary)
                segments.extend({**segment, "sub_task": label["sub_task"]}
                                for segment, label in zip(requested, value["labels"]["segments"]))
                previous_boundary = visual_boundary(requested[-1], value["visual_checks"][-1])
            if not segments:
                continue
            episode = episode_from(source, load_numeric(directory / "numeric"), Settings(**config["settings"]))
            states = np.asarray(episode["measured_states"])
        else:
            continue
        target = output / source["episode_id"].replace("/", "_")
        target.mkdir(exist_ok=True)
        episode_rows = []
        for segment in segments:
            start, end = segment["start_frame"], segment["end_frame"] + 1
            frames = sorted(set(np.linspace(start, end, 5, dtype=int).tolist()))
            row = {
                "episode_id": source["episode_id"], "task": source["task"],
                "segment_index": segment["segment_index"],
                "display_segment": segment["segment_index"] + 1,
                "start_frame": start, "target_frame": end,
                "left_action": segment["left_action"], "right_action": segment["right_action"],
                "sub_task": segment["sub_task"],
                "left": arm_metrics(states, start, end, 0),
                "right": arm_metrics(states, start, end, 8),
                "observations": [
                    {"frame": frame, "path": str(directory / "media/teacher" / f"{frame:07d}.jpg")}
                    for frame in frames
                ],
                "source_phases": [phase for phase in source["metadata"].get("action_config", [])
                                  if phase["start_frame"] < end and phase.get("end_frame", end) > start],
            }
            episode_rows.append(row)
        for sheet_index, first in enumerate(range(0, len(episode_rows), rows_per_sheet)):
            selected = episode_rows[first:first + rows_per_sheet]
            blocks = []
            for row in selected:
                wrapped = textwrap.wrap(row["sub_task"], width=205, break_long_words=False)
                height = 110 + 28 * len(wrapped) + 224
                block = Image.new("RGB", (3200, height), "#ffffff")
                draw = ImageDraw.Draw(block)
                draw.text((16, 8), f'{row["episode_id"]} | SEGMENT {row["display_segment"]} '
                          f'| frames {row["start_frame"]}-{row["target_frame"]} '
                          f'| L={row["left_action"]}, R={row["right_action"]}', fill="#000000", font=title_font)
                for index, line in enumerate(wrapped):
                    draw.text((16, 42 + index * 28), line, fill="#000000", font=font)
                y = 46 + 28 * len(wrapped)
                for arm in ("left", "right"):
                    metrics = row[arm]
                    draw.text((16, y), f'{arm.upper()} openness {metrics["start_openness"]}'
                              f' -> {metrics["target_openness"]}; range {metrics["openness_range"]}; '
                              f'XYZ path {metrics["position_path_m"]}m; delta {metrics["position_delta_m"]}',
                              fill="#000000", font=font)
                    y += 27
                for column, observation in enumerate(row["observations"]):
                    x = column * 640
                    draw.text((x + 8, y), f'observation {observation["frame"]}', fill="#000000", font=font)
                    with Image.open(observation["path"]) as image:
                        resized = image.copy()
                        resized.thumbnail((640, 192))
                        block.paste(resized, (x, y + 26))
                blocks.append(block)
            sheet = Image.new("RGB", (3200, sum(block.height for block in blocks) + 12 * len(blocks)), "#c8c8c8")
            y = 0
            for block in blocks:
                sheet.paste(block, (0, y))
                y += block.height + 12
            path = target / f"sheet_{sheet_index + 1:02d}.jpg"
            sheet.save(path, quality=95)
            for row in selected:
                row["evidence_sheet"] = str(path.resolve())
        rows.extend(episode_rows)
        print(source["episode_id"], len(episode_rows), "segments", flush=True)
    if not include_accepted_batches:
        assert len(rows) == sum(entry["segments"] for entry in accepted)
    atomic_json(output / "evidence.json", {
        "source_run": str(run), "source_version": config["version"],
        "source_export_hashes": {entry["episode_id"]: file_hash(Path(entry["path"]) / "vla.json")
                                 for entry in accepted},
        "episodes": len({row["episode_id"] for row in rows}), "segments": len(rows), "rows": rows,
        "partial_accepted_batches": include_accepted_batches,
        "camera_convention": "each montage is head | hand_left | hand_right; wrist views may show the other hand",
        "openness_convention": "measured proxy approximately 1=open and 0=closed; not proof of grasp",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-accepted-batches", action="store_true",
                        help="Temporary inspection only; includes accepted caches of unfinished episodes")
    args = parser.parse_args()
    build(args.run, args.output, include_accepted_batches=args.include_accepted_batches)


if __name__ == "__main__":
    main()
