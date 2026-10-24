#!/usr/bin/env python3
"""Prepare a fresh API pilot from verified numeric/video stages, never old labels."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.agibot.config import Settings
from atomic_seg.pretrain.agibot.contract import action_contract, student_system, teacher_system
from atomic_seg.pretrain.agibot.media import montage, sample_frames
from atomic_seg.pretrain.agibot.runner import episode_dir, implementation, stage_identity
from atomic_seg.pretrain.agibot.source import check_sources, iter_inventory
from atomic_seg.pretrain.realomni.storage import (
    atomic_bytes, atomic_json, commit_directory, digest, encode, read_json, staging, verify_directory,
)
from scripts.agibot import freeze


def prepare(baseline, output, selection):
    baseline, output = baseline.resolve(), output.resolve()
    if output.exists():
        raise ValueError("pilot output must be new; resume it using agibot.sh")
    ids = read_json(selection)
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("selection must contain unique episode identities")
    old = read_json(baseline / "run_config.json")
    settings = Settings(**old["settings"])
    value = {"schema": old["schema"], "settings": settings.as_dict(), "selection": sorted(ids),
             "implementation": implementation(), "action_contract": action_contract(settings),
             "student_system": student_system(settings), "teacher_system": teacher_system(settings)}
    if value["action_contract"] != old["action_contract"] or value["student_system"] != old["student_system"]:
        raise ValueError("pilot requires unchanged numerical and student contracts")
    allowed = {"pretrain/agibot/annotate.py", "pretrain/agibot/contract.py",
               "pretrain/agibot/media.py", "pretrain/agibot/api.py"}
    changed = {name for name in old["implementation"]["files"]
               if old["implementation"]["files"][name] != value["implementation"]["files"].get(name)}
    if changed - allowed or old["implementation"]["dependencies"] != value["implementation"]["dependencies"]:
        raise ValueError(f"non-annotation implementation changed: {sorted(changed - allowed)}")
    rows = [row for row in iter_inventory(baseline) if row["episode_id"] in ids]
    if {row["episode_id"] for row in rows} != set(ids):
        raise ValueError("baseline is missing selected episodes")
    old_inventory_identity = digest({"source": str(Path(settings.source_root).resolve()),
                                    "settings": settings.as_dict(), "selection": old["selection"]})
    verify_directory(baseline / "inventory", old_inventory_identity)
    version = digest(value)
    freeze(output)
    atomic_json(output / "run_config.json", {"version": version, **value})
    temporary = staging(output, "inventory")
    atomic_bytes(temporary / "episodes.jsonl", "".join(encode(row) + "\n" for row in rows).encode())
    atomic_bytes(temporary / "index.jsonl", "".join(encode({k: row[k] for k in (
        "episode_id", "key", "source_identity", "split", "length")}) + "\n" for row in rows).encode())
    atomic_json(temporary / "summary.json", {"episodes": len(rows),
                "measured_frames": sum(row["length"] for row in rows),
                "tasks": len({row["task_dir"] for row in rows})})
    identity = digest({"source": str(Path(settings.source_root).resolve()), "settings": settings.as_dict(),
                       "selection": sorted(ids)})
    commit_directory(temporary, output / "inventory", identity)
    reused = []
    for row in rows:
        check_sources(row)
        before, after = episode_dir(baseline, row), episode_dir(output, row)
        stages = ["numeric", "checked", "segmented"] + ["camera_" + c for c in settings.cameras]
        for stage in stages:
            commit = verify_directory(before / stage, stage_identity(old["version"], row, stage))
            temporary = staging(after, stage)
            for item in commit["files"]:
                target = temporary / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(before / stage / item["path"], target)
            commit_directory(temporary, after / stage, stage_identity(version, row, stage))
        verify_directory(before / "media", stage_identity(old["version"], row, "media"))
        images = read_json(before / "media/images.json")
        segments = read_json(before / "segmented/segmentation.json")["segments"]
        temporary = staging(after, "media")
        for camera in settings.student_cameras:
            for relative in images[camera].values():
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(before / "media" / relative, target)
        (temporary / "teacher").mkdir()
        for frame in sample_frames(segments):
            montage([temporary / images[c][str(frame)] for c in settings.student_cameras],
                    settings.student_cameras, temporary / "teacher" / f"{frame:07d}.jpg")
        atomic_json(temporary / "images.json", images)
        commit_directory(temporary, after / "media", stage_identity(version, row, "media"))
        reused.append({"episode_id": row["episode_id"], "source_identity": row["source_identity"],
                       "stages": stages, "student_images": "byte-identical", "teacher_montages": "rebuilt"})
        print("Prepared fresh labels", row["episode_id"], len(segments), flush=True)
    atomic_json(output / "PILOT_PROVENANCE.json", {"baseline": str(baseline), "baseline_version": old["version"],
                "new_version": version, "reused_verified_stages": reused,
                "old_labels_reused": False, "boundaries_and_numeric_actions": "unchanged",
                "implementation_changed": sorted(changed)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.baseline, args.output, args.selection))


if __name__ == "__main__":
    main()
