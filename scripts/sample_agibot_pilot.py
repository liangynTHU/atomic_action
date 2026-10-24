#!/usr/bin/env python3
"""Sample unseen AgiBot tasks without looking at images or annotation outputs."""

import argparse
import json
from pathlib import Path
import random
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.agibot.config import SOURCE_ROOT
from atomic_seg.pretrain.agibot.config import Settings
from atomic_seg.pretrain.agibot.quality import inspect_numeric
from atomic_seg.pretrain.agibot.source import NumericReader, rows_for_task
from atomic_seg.pretrain.realomni.storage import atomic_json, file_hash


def sample(source, output, exclusions, seed, count, candidates_per_task, numeric_gate=False):
    if output.exists():
        raise ValueError("sampling output must be a new directory")
    excluded = set()
    provenance = []
    for path in exclusions:
        value = json.loads(path.read_text())
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("exclusions must be episode selection lists")
        excluded.update(item.split("/")[0] for item in value)
        provenance.append({"path": str(path.resolve()), "sha256": file_hash(path)})
    tasks = [path for path in sorted(source.glob("task_*"))
             if path.is_dir() and path.name not in excluded]
    if len(tasks) < count:
        raise ValueError("not enough unseen tasks")
    rng = random.Random(seed)
    rng.shuffle(tasks)
    selected, rows, accepted = [], [], []
    settings, reader = Settings(source_root=str(source.resolve())), NumericReader()
    output.mkdir(parents=True)
    for task in tasks if numeric_gate else tasks[:count]:
        episodes = []
        for path in sorted((task / "meta/episodes").glob("chunk-*/*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "length"])
            episodes.extend(table.to_pylist())
        if not episodes or len({item["episode_index"] for item in episodes}) != len(episodes):
            raise ValueError("empty or duplicate episode metadata")
        rng.shuffle(episodes)
        candidates = episodes[:candidates_per_task]
        ids = [f"{task.name}/episode_{item['episode_index']:06d}" for item in candidates]
        result = {"task": task.name, "eligible_metadata_episodes": len(episodes),
                     "candidate_order": ids,
                     "candidate_lengths": [item["length"] for item in candidates]}
        selected.extend(ids)
        if numeric_gate:
            source_rows = {row["episode_id"]: row for row in rows_for_task(task, settings)
                           if row["episode_id"] in ids}
            result["numeric_decisions"] = []
            for episode_id in ids:
                row = source_rows[episode_id]
                audit = inspect_numeric(row, reader.read(row), settings)
                result["numeric_decisions"].append({
                    "episode_id": episode_id, "passed": not audit["reasons"],
                    "reasons": audit["reasons"],
                })
                if not audit["reasons"]:
                    accepted.append(episode_id)
                    break
            print(task.name, result["numeric_decisions"], flush=True)
        else:
            print(task.name, ids, flush=True)
        rows.append(result)
        atomic_json(output / "TASK_DECISIONS.json", rows)
        if numeric_gate and len(accepted) == count:
            break
    if numeric_gate:
        atomic_json(output / "numeric_passed.json", accepted)
        if len(accepted) != count:
            raise ValueError(f"only {len(accepted)} unseen tasks passed the numeric gate")
    atomic_json(output / "candidates.json", selected)
    atomic_json(output / "SAMPLING.json", {
        "seed": seed, "source": str(source.resolve()), "task_population": len(tasks),
        "excluded_tasks": sorted(excluded), "exclusion_inputs": provenance,
        "method": "uniform random unseen tasks, then uniform random episode order within each task",
        "selection_policy": "first source-gate-passing episode per task in recorded candidate order; "
                            "never select by annotation quality or visual simplicity",
        "candidates_per_task": candidates_per_task, "tasks": rows,
        "is_episode_weighted_population_sample": False,
        "images_or_labels_inspected_for_sampling": False,
        "numeric_gate": numeric_gate,
    })
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_ROOT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--candidates-per-task", type=int, default=2)
    parser.add_argument("--numeric-gate", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.candidates_per_task < 1:
        parser.error("counts must be positive")
    sample(args.source, args.output, args.exclude, args.seed, args.count, args.candidates_per_task,
           args.numeric_gate)


if __name__ == "__main__":
    main()
