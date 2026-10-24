#!/usr/bin/env python3
"""Create and verify two immutable, balanced AgiBot episode selections."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.agibot.config import SOURCE_ROOT
from atomic_seg.pretrain.realomni.storage import (
    atomic_json,
    commit_directory,
    digest,
    fingerprint,
    lock,
    read_json,
    staging,
    verify_directory,
)

ALGORITHM = "numeric_task_episode_order_alternating_v1"


def default_split_dir(source):
    source_key = hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:16]
    return ROOT / ".runtime/agibot_split2" / source_key


def source_catalog(source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dataset directory is not accessible: {source}")
    tasks = sorted(
        (path for path in source.glob("task_*") if path.is_dir()),
        key=lambda path: (int(path.name[5:]) if path.name[5:].isdigit() else sys.maxsize, path.name),
    )
    if not tasks:
        raise ValueError(f"no task_* directories found: {source}")
    catalog = []
    for task in tasks:
        if not re.fullmatch(r"task_[0-9]+", task.name):
            raise ValueError(f"invalid task directory: {task}")
        info = task / "meta/info.json"
        shards = sorted((task / "meta/episodes").glob("chunk-*/*.parquet"))
        if not shards:
            raise ValueError(f"no episode Parquet indexes found: {task}")
        catalog.append({
            "task": task.name,
            "info": fingerprint(info),
            "shards": [fingerprint(path) for path in shards],
        })
    return catalog


def read_episodes(catalog):
    import pyarrow.parquet as pq

    episodes = []
    tasks = []
    for item in catalog:
        info = read_json(item["info"]["path"])
        if info.get("codebase_version") != "v3.0" or info.get("robot_type") != "a2d" or info.get("fps") != 30:
            raise ValueError(f"unsupported AgiBot task schema: {item['task']}")
        indices = set()
        for shard in item["shards"]:
            parquet = pq.ParquetFile(shard["path"])
            for batch in parquet.iter_batches(batch_size=4096, columns=["episode_index"]):
                for value in batch.column(0).to_pylist():
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"invalid episode index in {shard['path']}: {value}")
                    if value in indices:
                        raise ValueError(f"duplicate episode: {item['task']}/episode_{value:06d}")
                    indices.add(value)
        if len(indices) != info.get("total_episodes"):
            raise ValueError(f"episode count differs from info.json: {item['task']}")
        episodes.extend(f"{item['task']}/episode_{index:06d}" for index in sorted(indices))
        tasks.append({"task": item["task"], "episodes": len(indices)})
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for two nonempty selections")
    return episodes, tasks


def verify_selections(target, identity, catalog):
    verify_directory(target, identity)
    summary = read_json(target / "summary.json")
    if summary["source_catalog"] != catalog or summary["algorithm"] != ALGORITHM:
        raise ValueError("split metadata does not match the source or allocation algorithm")
    groups = [read_json(target / f"part{part}.json") for part in (1, 2)]
    for group in groups:
        if (not isinstance(group, list) or not group
                or any(not isinstance(episode, str) or not re.fullmatch(r"task_[0-9]+/episode_[0-9]{6,}", episode)
                       for episode in group)):
            raise ValueError("invalid episode selection")
    sets = [set(group) for group in groups]
    if any(len(group) != len(unique) for group, unique in zip(groups, sets)):
        raise ValueError("duplicate episode within a partition")
    if sets[0] & sets[1]:
        raise ValueError("episode partitions overlap")
    counts = [len(group) for group in groups]
    if counts != summary["part_counts"] or sum(counts) != summary["total_episodes"] or abs(counts[0] - counts[1]) > 1:
        raise ValueError("episode partitions are incomplete or not balanced")
    combined = sets[0] | sets[1]
    if digest(sorted(combined)) != summary["episode_ids_digest"]:
        raise ValueError("episode coverage differs from the committed source inventory")
    return summary


def prepare(source, split_dir=None):
    source = Path(source).resolve()
    split_dir = Path(split_dir).resolve() if split_dir else default_split_dir(source)
    target = split_dir / "selection"
    with lock(split_dir / "split.lock", blocking=True):
        catalog = source_catalog(source)
        identity = digest({"source": str(source), "algorithm": ALGORITHM, "catalog": catalog})
        if target.exists():
            commit = read_json(target / "COMMIT.json")
            if commit["identity"] != identity:
                raise ValueError(
                    "source episode metadata changed; keep the existing selections for old runs "
                    "and set a new AGIBOT_SPLIT_DIR and AGIBOT_OUTPUT for a new run"
                )
            summary = verify_selections(target, identity, catalog)
        else:
            episodes, tasks = read_episodes(catalog)
            if source_catalog(source) != catalog:
                raise RuntimeError("source episode metadata changed while splitting; retry after the dataset is stable")
            groups = [episodes[::2], episodes[1::2]]
            part_sets = [set(group) for group in groups]
            for task in tasks:
                prefix = task["task"] + "/"
                task["part_counts"] = [sum(episode.startswith(prefix) for episode in group) for group in part_sets]
            summary = {
                "source_root": str(source),
                "algorithm": ALGORITHM,
                "total_tasks": len(tasks),
                "total_episodes": len(episodes),
                "part_counts": [len(group) for group in groups],
                "episode_ids_digest": digest(sorted(episodes)),
                "source_catalog": catalog,
                "tasks": tasks,
            }
            temporary = staging(split_dir, "selection")
            for part, group in enumerate(groups, 1):
                atomic_json(temporary / f"part{part}.json", group)
            atomic_json(temporary / "summary.json", summary)
            commit_directory(temporary, target, identity)
            summary = verify_selections(target, identity, catalog)
    return {
        "source_root": str(source),
        "total_tasks": summary["total_tasks"],
        "total_episodes": summary["total_episodes"],
        "part_counts": summary["part_counts"],
        "selection_dir": str(target),
        "part1_selection": str(target / "part1.json"),
        "part2_selection": str(target / "part2.json"),
        "coverage": "all source episodes, no duplicates or overlap",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_ROOT)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--print-default-dir", action="store_true")
    args = parser.parse_args()
    if args.print_default_dir:
        print(default_split_dir(args.source))
        return
    print(json.dumps(prepare(args.source, args.split_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"agibot split: {error}", file=sys.stderr)
        raise SystemExit(2)
