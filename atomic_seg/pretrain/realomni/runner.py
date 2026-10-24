"""Bounded, durable whole-episode inspection and atomic segmentation."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .quality import BASE_POLICY, inspect_episode
from .reference import ACTION_CONTRACT, reference_episode, segment_reference
from .storage import (
    Journal, atomic_json, check_source, commit_directory, digest, encode,
    episode_dir, file_hash, fingerprint, iter_jsonl, lock, process_identity,
    read_json, staging, status, sync_directory, verify_directory,
)


def implementation_identity():
    package = Path(__file__).resolve().parents[2]
    # Include shared numerical algorithms and prompts, not Git HEAD: this
    # workspace contains intentional uncommitted implementation changes.
    names = ["config.py", "segmentation.py", "features.py", "geometry.py", "still_policy.py",
             "generation.py", "action_language.py", "prompt_contract.py", "common/numeric.py",
             "pretrain/abc_still.py", "pretrain/length_policy.py", "pretrain/segmentation.py",
             "pretrain/prompts.py", "pretrain/student_format.py"]
    files = {name: file_hash(package / name) for name in names}
    files.update({str(p.relative_to(package)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    dependencies = {}
    for name in ("numpy", "scipy", "av", "mcap", "mcap-protobuf-support", "protobuf"):
        dependencies[name] = importlib.metadata.version(name)
    return {"files": files, "dependencies": dependencies}


def initialize_run(run, settings, *, selection=None):
    run = Path(run).resolve()
    source = Path(settings.source_root).resolve()
    if not source.is_dir() or run == source or source in run.parents or run in source.parents:
        raise ValueError("output must be separate from the existing source root")
    config = {"schema": "realomni_whole_episode_v1", "settings": settings.as_dict(),
              "implementation": implementation_identity(), "action_contract": ACTION_CONTRACT,
              "base_policy": BASE_POLICY, "selection": sorted(selection) if selection is not None else None}
    version = digest(config)
    record = {"version": version, **config}
    path = run / "run_config.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("code, settings or source selection changed; use a new output directory")
    else:
        atomic_json(path, record)
    return version


def inventory(run, settings, *, selection=None):
    """Commit the complete selected inventory before touching any episode."""
    run, root = Path(run), Path(settings.source_root).resolve()
    destination = run / "inventory"
    identity = digest({"root": str(root), "selection": sorted(selection) if selection is not None else None,
                       "logical_shards": settings.logical_shards})
    if destination.exists():
        verify_directory(destination, identity)
        return destination / "episodes.jsonl"
    for leftover in run.glob(".pending_inventory_*"):
        shutil.rmtree(leftover)
    temporary = staging(run, "inventory")

    def paths():
        if selection is not None:
            if not selection or len(set(selection)) != len(selection):
                raise ValueError("episode selection must be nonempty and unique")
            for relative in sorted(selection):
                p = (root / relative).resolve()
                if (root not in p.parents or p.suffix != ".mcap" or not p.is_file()
                        or relative != str(p.relative_to(root))):
                    raise ValueError(f"selection is not a source MCAP file: {relative}")
                yield p
        else:
            def fail(error):
                raise error

            for parent, directories, filenames in os.walk(root, onerror=fail):
                directories[:] = sorted(d for d in directories if not d.startswith("."))
                for name in sorted(filenames):
                    if name.endswith(".mcap"):
                        yield Path(parent) / name

    count = 0
    with (temporary / "episodes.jsonl").open("w") as out:
        for path in paths():
            episode_id = str(path.relative_to(root))
            key = digest(episode_id)
            source_file = fingerprint(path)
            row = {"episode_id": episode_id, "key": key, "file": source_file,
                   "source_identity": digest(source_file), "shard": int(key[:16], 16) % settings.logical_shards}
            out.write(encode(row) + "\n")
            count += 1
        out.flush()
        os.fsync(out.fileno())
    atomic_json(temporary / "summary.json", {"episodes": count, "scope": "selection" if selection is not None else "source_root"})
    commit_directory(temporary, destination, identity)
    return destination / "episodes.jsonl"


def save_episode(directory, episode):
    arrays = {"measured_states": np.asarray(episode["measured_states"], dtype=np.float64),
              "measured_timestamps_ns": np.asarray(episode["measured_timestamps_ns"], dtype=np.int64)}
    numeric_topics = {}
    for i, (topic, stream) in enumerate(episode.get("native_aux", {}).items()):
        numeric_topics[topic] = {name: f"source_{i}_{name}" for name in stream}
        arrays.update({numeric_topics[topic][name]: np.asarray(values) for name, values in stream.items()})
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    metadata = {k: episode[k] for k in ("id", "task", "source", "alignment", "camera_source_indices")}
    metadata.update(native_array_keys=numeric_topics, action_contract=ACTION_CONTRACT,
                    terminal_observation_retained=True)
    if episode.get("crop"):
        metadata["crop"] = episode["crop"]
    atomic_json(directory / "episode.json", metadata)


def load_episode(directory):
    metadata = read_json(directory / "episode.json")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        # The persisted states already have normalized, sign-continuous quaternions.
        measured = data["measured_states"].copy()
        episode = reference_episode(measured, data["measured_timestamps_ns"], task=metadata["task"],
                                    episode_id=metadata["id"], source=metadata["source"])
        # Preserve the exact persisted floating-point representation on restart.
        episode.update(measured_states=measured.tolist(), states=measured[:-1].tolist(),
                       actions=measured[1:].tolist(), command_endpoints=measured[1:].tolist())
    episode.update(alignment=metadata["alignment"], camera_source_indices=metadata["camera_source_indices"])
    if metadata.get("crop"):
        episode["crop"] = metadata["crop"]
    return episode


def process_episode(run, row, settings, version):
    run = Path(run)
    directory = episode_dir(run, row)
    with lock(directory / "episode.lock"):
        return _process_locked(run, row, settings, version)


def _process_locked(run, row, settings, version):
    directory = episode_dir(run, row)
    checks, segmented = directory / "checked", directory / "segmented"
    check_identity = digest({"version": version, "source": row["source_identity"], "stage": "checked"})
    segment_identity = digest({"version": version, "source": row["source_identity"], "stage": "segmented"})
    journal = Journal(directory / "events.jsonl")
    old = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
    if old and (old["version"] != version or old["source_identity"] != row["source_identity"]):
        raise ValueError("episode checkpoint identity changed")
    attempts = old.get("attempts", 0)

    def record(state, **details):
        return status(run, row, version, state, attempts=attempts, owner=process_identity(),
                      **{"stage": state, "retryable": False, **details})

    try:
        check_source(row)
        for leftover in directory.glob(".pending_*"):
            shutil.rmtree(leftover)
        episode = None
        if checks.exists():
            verify_directory(checks, check_identity)
            saved = read_json(checks / "decision.json")
            # Stat alone is insufficient if a source has been replaced while
            # preserving its size and mtime.
            if file_hash(row["file"]["path"]) != saved["source_sha256"]:
                raise RuntimeError("source content changed since quality decision")
            if saved["decision"] == "rejected":
                return record("rejected", quality="rejected", reasons=saved["audit"]["reasons"],
                              decision_path=str(checks / "decision.json"), resumed=True, error=None)
        else:
            attempts += 1
            record("checking", quality="pending", error=None, resumed=bool(old))
            journal.append({"event": "inspection_started", "attempt": attempts, "time": time.time()})
            source_sha256 = file_hash(row["file"]["path"])

            def checkpoint(stage, details):
                atomic_json(directory / "inspection.json", {"stage": stage, "version": version,
                            "source_identity": row["source_identity"], "details": details, "time": time.time()})
                record("checking", stage=stage)

            audit, episode = inspect_episode(row["file"]["path"], settings, checkpoint=checkpoint)
            if not audit.get("reasons") and episode is None:
                raise ValueError("inspection passed without a complete measured trajectory")
            check_source(row)
            if file_hash(row["file"]["path"]) != source_sha256:
                raise RuntimeError("source content changed during inspection")
            decision = "rejected" if audit["reasons"] else "accepted"
            temporary = staging(directory, "checked")
            saved = {"decision": decision, "source_sha256": source_sha256, "audit": audit}
            atomic_json(temporary / "decision.json", saved)
            if decision == "accepted":
                episode.update(id=row["key"], source={"file": row["file"]["path"],
                               "sha256": source_sha256, "episode_id": row["episode_id"], "split": "unspecified"})
                # Directory labels are coarse source context, not generated subtask annotations.
                episode["task"] = str(Path(row["episode_id"]).parent)
                save_episode(temporary, episode)
            commit_directory(temporary, checks, check_identity)
            journal.append({"event": "quality_committed", "decision": decision, "time": time.time()})
            if decision == "rejected":
                return record("rejected", quality="rejected", reasons=audit["reasons"],
                              decision_path=str(checks / "decision.json"), error=None)
        record("checked", quality="accepted", decision_path=str(checks / "decision.json"), error=None)
        if segmented.exists():
            verify_directory(segmented, segment_identity)
            result = read_json(segmented / "segmentation.json")
            return record("segmented", segments=len(result["segments"]),
                          reference_action_rows=result["reference_action_rows"],
                          segmentation_path=str(segmented / "segmentation.json"), resumed=True)
        if episode is None:
            episode = load_episode(checks)
        record("segmenting")
        result = segment_reference(episode, length_policy=settings.length_policy)
        temporary = staging(directory, "segmented")
        atomic_json(temporary / "segmentation.json", result)
        check_source(row)
        commit_directory(temporary, segmented, segment_identity)
        journal.append({"event": "segmentation_committed", "segments": len(result["segments"]), "time": time.time()})
        return record("segmented", segments=len(result["segments"]),
                      reference_action_rows=result["reference_action_rows"],
                      segmentation_path=str(segmented / "segmentation.json"), error=None)
    except Exception as error:
        journal.append({"event": "error", "type": type(error).__name__, "detail": str(error), "time": time.time()})
        return record("error", error={"type": type(error).__name__, "detail": str(error)}, retryable=True)


def write_progress(run, manifest):
    """Rebuild from per-episode checkpoints; a torn aggregate loses no decisions."""
    run = Path(run)
    counts, reasons = Counter(), Counter()
    total_rows = total_segments = 0
    temporary = run / "episode_status.jsonl.tmp"
    with temporary.open("w") as out:
        for row in iter_jsonl(manifest):
            path = episode_dir(run, row) / "status.json"
            state = read_json(path) if path.exists() else {"status": "pending"}
            counts[state["status"]] += 1
            if state["status"] == "segmented":
                total_rows += state["reference_action_rows"]
                total_segments += state["segments"]
            if state["status"] == "rejected":
                reasons.update({r["code"] for r in state["reasons"]})
            out.write(encode({"episode_id": row["episode_id"], "key": row["key"],
                              "source_file": row["file"]["path"], **state}) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, run / "episode_status.jsonl")
    sync_directory(run)
    report = {"updated_at": time.time(), "discovered": sum(counts.values()), "counts": dict(counts),
              "rejection_episode_counts": dict(reasons), "reference_action_rows": total_rows,
              "segments": total_segments, "base_policy": BASE_POLICY,
              "scope": read_json(Path(manifest).with_name("summary.json"))["scope"]}
    atomic_json(run / "progress.json", report)
    return report


def run_preparation(run, settings, *, selection=None, max_episodes=None):
    """One writer, bounded episode memory; repeat the same command to resume."""
    run = Path(run).resolve()
    source = Path(settings.source_root).resolve()
    if not source.is_dir() or run == source or source in run.parents or run in source.parents:
        raise ValueError("output must be separate from the existing source root")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    with lock(run / "run.lock"):
        version = initialize_run(run, settings, selection=selection)
        manifest = inventory(run, settings, selection=selection)
        write_progress(run, manifest)
        count, updated = 0, time.monotonic()
        for row in iter_jsonl(manifest):
            # Budget counts unfinished episodes, so repeated limited invocations
            # progress through the inventory rather than redoing its prefix.
            path = episode_dir(run, row) / "status.json"
            terminal = path.exists() and read_json(path)["status"] in {"rejected", "segmented"}
            if not terminal and max_episodes is not None and count >= max_episodes:
                continue
            result = process_episode(run, row, settings, version)
            if not terminal:
                count += 1
            print(encode({"episode_id": row["episode_id"], "status": result["status"]}), flush=True)
            if time.monotonic() - updated >= 30:
                write_progress(run, manifest)
                updated = time.monotonic()
        return write_progress(run, manifest)
