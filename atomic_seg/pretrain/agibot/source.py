"""Stream LeRobot v3 episode indexes and read bounded Parquet row groups."""

from pathlib import Path
import json
import os

import numpy as np
import pyarrow.parquet as pq

from atomic_seg.pretrain.realomni.storage import (
    atomic_json, commit_directory, digest, encode, file_hash, fingerprint,
    iter_jsonl, lock, read_json, staging, verify_directory,
)


def optional_identity(path):
    path = Path(path)
    return fingerprint(path) if path.is_file() else {"path": str(path.resolve()), "missing": True}


def check_sources(row):
    for expected in row["source_files"]:
        if optional_identity(expected["path"]) != expected:
            raise RuntimeError("source changed; use a new version: " + expected["path"])


def data_catalog(task):
    result = []
    for path in sorted((task / "data").glob("chunk-*/*.parquet")):
        try:
            pf = pq.ParquetFile(path)
            index = pf.schema.names.index("episode_index")
            groups = []
            for i in range(pf.num_row_groups):
                stats = pf.metadata.row_group(i).column(index).statistics
                if stats is None or not stats.has_min_max:
                    values = pf.read_row_group(i, columns=["episode_index"])["episode_index"].to_numpy()
                    lo, hi = int(values.min()), int(values.max())
                else:
                    lo, hi = int(stats.min), int(stats.max)
                groups.append({"row_group": i, "episode_min": lo, "episode_max": hi})
            result.append({"path": str(path.resolve()), "groups": groups, "identity": optional_identity(path)})
        except ValueError as error:
            result.append({"path": str(path.resolve()), "groups": [], "error": str(error)})
    return result


def rows_for_task(task, settings):
    info = read_json(task / "meta/info.json")
    if info.get("codebase_version") != "v3.0" or info.get("robot_type") != "a2d" or info.get("fps") != settings.fps:
        raise ValueError(f"unsupported task schema: {task}")
    tasks = {int(r["task_index"]): r["task"] for r in pq.read_table(task / "meta/tasks.parquet").to_pylist()}
    catalog = data_catalog(task)
    common = [fingerprint(task / "meta/info.json"), fingerprint(task / "meta/tasks.parquet")]
    seen = set()
    count = 0
    for meta in sorted((task / "meta/episodes").glob("chunk-*/*.parquet")):
        pf = pq.ParquetFile(meta)
        columns = [name for name in pf.schema_arrow.names if not name.startswith("stats/")]
        for batch in pf.iter_batches(batch_size=512, columns=columns):
            for record in batch.to_pylist():
                ep = int(record["episode_index"])
                if ep in seen:
                    raise ValueError(f"duplicate episode identity: {task.name}/{ep}")
                seen.add(ep)
                count += 1
                episode_id = f"{task.name}/episode_{ep:06d}"
                data = []
                for item in catalog:
                    groups = [g["row_group"] for g in item["groups"] if g["episode_min"] <= ep <= g["episode_max"]]
                    if groups:
                        data.append({"path": item["path"], "row_groups": groups})
                videos = {}
                for camera in settings.cameras:
                    key = "observation.images." + camera
                    prefix = "videos/" + key
                    try:
                        relative = info["video_path"].format(video_key=key,
                            chunk_index=record[prefix + "/chunk_index"], file_index=record[prefix + "/file_index"])
                        videos[camera] = {"path": str((task / relative).resolve()),
                            "from_s": record[prefix + "/from_timestamp"], "to_s": record[prefix + "/to_timestamp"]}
                    except KeyError:
                        videos[camera] = {"missing_metadata": True}
                source_files = common + [fingerprint(meta)]
                source_files += [optional_identity(item["path"]) for item in data]
                source_files += [optional_identity(v["path"]) for v in videos.values() if "path" in v]
                # Mutable auxiliary annotation files are not read by this run.
                # Their publication must not invalidate immutable source tracks.
                split = None
                for name, span in info.get("splits", {}).items():
                    lo, hi = [int(x) for x in span.split(":")]
                    if lo <= ep < hi:
                        if split is not None:
                            raise ValueError("overlapping source splits")
                        split = name
                row = {"episode_id": episode_id, "key": digest(episode_id),
                    "task_dir": str(task.resolve()),
                    "episode_index": ep, "length": int(record["length"]), "split": split or "unspecified",
                    "metadata": record, "features": {k: v for k, v in info["features"].items() if v["dtype"] != "video"},
                    "task": " | ".join(record.get("tasks") or list(tasks.values())),
                    "data": data, "videos": videos, "source_files": source_files,
                    "catalog_errors": [{"path": x["path"], "error": x["error"]} for x in catalog if "error" in x]}
                row["source_identity"] = digest(row)
                yield row
    if count != info["total_episodes"]:
        raise ValueError(f"episode index count differs from info.json: {task}")


def inventory(run, settings, selection=None):
    run = Path(run)
    identity = digest({"source": str(Path(settings.source_root).resolve()), "settings": settings.as_dict(),
                       "selection": sorted(selection) if selection else None})
    target = run / "inventory"
    with lock(run / "control/inventory.lock", blocking=True):
        if target.exists():
            verify_directory(target, identity)
            return target
        temporary = staging(run, "inventory")
        selected = set(selection or [])
        found = set()
        counts = {"episodes": 0, "measured_frames": 0, "tasks": 0}
        from contextlib import ExitStack
        with ExitStack() as stack:
            stream = stack.enter_context((temporary / "episodes.jsonl").open("w"))
            index = stack.enter_context((temporary / "index.jsonl").open("w"))
            for task in sorted(Path(settings.source_root).glob("task_*")):
                if selected and not any(s.startswith(task.name + "/") for s in selected):
                    continue
                counts["tasks"] += 1
                for row in rows_for_task(task, settings):
                    if selected and row["episode_id"] not in selected:
                        continue
                    found.add(row["episode_id"])
                    counts["episodes"] += 1
                    counts["measured_frames"] += row["length"]
                    stream.write(encode(row) + "\n")
                    index.write(encode({k: row[k] for k in (
                        "episode_id", "key", "source_identity", "split", "length")}) + "\n")
                print("inventory", task.name, counts["episodes"], flush=True)
            for handle in (stream, index):
                handle.flush()
                os.fsync(handle.fileno())
        if not counts["episodes"] or (selected and found != selected):
            raise ValueError(f"empty/missing episode selection: {sorted(selected - found)}")
        atomic_json(temporary / "summary.json", counts)
        commit_directory(temporary, target, identity)
    return target


def iter_inventory(run):
    directory = Path(run) / "inventory"
    if (directory / "episodes.jsonl").is_file():
        yield from iter_jsonl(directory / "episodes.jsonl")
    else:
        # Read-only support for reviews of already generated legacy data.
        for path in sorted(directory.glob("shard_*.jsonl")):
            yield from iter_jsonl(path)


def iter_index(run):
    yield from iter_jsonl(Path(run) / "inventory/index.jsonl")


class NumericReader:
    """One cached row group per worker; shared files are not reread per frame."""

    def __init__(self):
        self.key = None
        self.table = None

    def read(self, row):
        import pyarrow as pa
        import pyarrow.compute as pc

        pieces = []
        for item in row["data"]:
            pf = pq.ParquetFile(item["path"])
            for group in item["row_groups"]:
                key = (item["path"], group)
                if key != self.key:
                    self.table = pf.read_row_group(group)
                    self.key = key
                part = self.table.filter(pc.equal(self.table["episode_index"], row["episode_index"]))
                if len(part):
                    pieces.append(part)
        if not pieces:
            return {}
        table = pa.concat_tables(pieces)
        return {k: np.asarray(v) for k, v in table.to_pydict().items()}
