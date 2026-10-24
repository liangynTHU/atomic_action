#!/usr/bin/env python3
"""Verify paired AgiBot exports differ only in annotation text and run paths."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.realomni.storage import (
    atomic_json, file_hash, read_json, verify_directory,
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def without_label(segment):
    return {key: value for key, value in segment.items() if key != "sub_task"}


def student_message(message):
    result = dict(message)
    if message["role"] == "assistant":
        result["content"] = without_label(json.loads(message["content"]))
    elif message["role"] == "user":
        lines = message["content"].splitlines()
        for index, line in enumerate(lines):
            prefix = "Previous command: "
            if line.startswith(prefix):
                lines[index] = prefix + json.dumps(
                    without_label(json.loads(line[len(prefix):])), sort_keys=True)
        result["content"] = "\n".join(lines)
    return result


def verify(baseline, run, output):
    baseline, run, output = baseline.resolve(), run.resolve(), output.resolve()
    for source in (baseline, run):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("verification report must not overlap either run")
    old_config, new_config = read_json(baseline / "run_config.json"), read_json(run / "run_config.json")
    for key in ("action_contract", "student_system", "settings"):
        if old_config[key] != new_config[key]:
            raise ValueError(f"paired {key} changed")
    old_entries = {entry["episode_id"]: entry for entry in jsonl(baseline / "accepted.jsonl")}
    new_entries = jsonl(run / "accepted.jsonl")
    expected = new_config["selection"]
    if (not expected or len(new_entries) != len(expected)
            or {entry["episode_id"] for entry in new_entries} != set(expected)):
        raise ValueError("paired run must have every selected episode committed exactly once")
    results = []
    for entry in new_entries:
        previous = old_entries[entry["episode_id"]]
        if previous["segments"] != entry["segments"] or previous["split"] != entry["split"]:
            raise ValueError("paired segment count or split changed")
        old, new = Path(previous["path"]), Path(entry["path"])
        for path in (old, new):
            verify_directory(path, read_json(path / "COMMIT.json")["identity"])
        old_vla, new_vla = read_json(old / "vla.json"), read_json(new / "vla.json")
        for value in (old_vla, new_vla):
            value["segments"] = [without_label(segment) for segment in value["segments"]]
        if file_hash(Path(old_vla.pop("native_arrays"))) != file_hash(Path(new_vla.pop("native_arrays"))):
            raise ValueError("paired native numerical arrays changed")
        if old_vla != new_vla:
            raise ValueError("paired segmentation, guide, chunks or source changed")
        with np.load(old / "reference.npz", allow_pickle=False) as before, np.load(
                new / "reference.npz", allow_pickle=False) as after:
            if set(before.files) != set(after.files):
                raise ValueError("paired reference array fields changed")
            for name in before.files:
                if (before[name].dtype != after[name].dtype
                        or not np.array_equal(before[name], after[name])):
                    raise ValueError(f"paired reference array changed: {name}")
        old_meta, new_meta = jsonl(old / "metadata.jsonl"), jsonl(new / "metadata.jsonl")
        for rows in (old_meta, new_meta):
            for row in rows:
                row["segment"] = without_label(row["segment"])
        if old_meta != new_meta:
            raise ValueError("paired metadata or causal image roles changed")
        old_rows, new_rows = jsonl(old / "vlm.jsonl"), jsonl(new / "vlm.jsonl")
        if len(old_rows) != len(new_rows) or len(new_rows) != entry["segments"]:
            raise ValueError("paired training row count changed")
        image_count = 0
        for before, after in zip(old_rows, new_rows):
            if set(before) != {"messages", "images"} or set(after) != set(before):
                raise ValueError("paired training JSON shape changed")
            if [student_message(m) for m in before["messages"]] != [
                    student_message(m) for m in after["messages"]]:
                raise ValueError("paired student messages changed beyond sub_task text")
            if len(before["images"]) != len(after["images"]):
                raise ValueError("paired image count changed")
            for first, second in zip(before["images"], after["images"]):
                if file_hash(Path(first)) != file_hash(Path(second)):
                    raise ValueError("paired student image bytes changed")
                image_count += 1
        results.append({"episode_id": entry["episode_id"], "segments": entry["segments"],
                        "student_image_references_verified": image_count})
    report = {
        "passed": True, "baseline": str(baseline), "run": str(run),
        "baseline_version": old_config["version"], "new_version": new_config["version"],
        "episodes": len(results), "segments": sum(row["segments"] for row in results),
        "allowed_changes": ["sub_task", "Previous command sub_task", "run-specific absolute paths"],
        "verified_unchanged": ["source and split", "all segmentation fields and boundaries",
                               "guide_action and num_chunks", "native and reference numerical arrays",
                               "student system and observations", "student image bytes and order",
                               "training JSON format", "API and segmentation settings"],
        "per_episode": results,
    }
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.baseline, args.run, args.output)


if __name__ == "__main__":
    main()
