"""Bounded local API workers, durable stages and whole-episode export."""

from pathlib import Path
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib.metadata
import json
import math
import os
import shutil
import threading
import time

import numpy as np

from atomic_seg.pretrain.realomni.storage import (
    Journal, atomic_json, atomic_bytes, commit_directory, digest, encode, file_hash,
    lock, process_identity, read_json, staging, verify_directory,
)
from atomic_seg.pretrain.segmentation import target_for
from atomic_seg.pretrain.length_policy import validate_action_primitives
from .config import Settings
from .contract import action_contract, build_reference, build_user, segment, student_system, teacher_system
from .source import NumericReader, check_sources, inventory, iter_inventory, iter_index
from .quality import inspect_numeric
from .media import InvalidVideo, montage, read_video, sample_frames
from .annotate import AnnotationRejected, ServiceUnavailable, annotate


def implementation():
    package = Path(__file__).resolve().parents[2]
    names = ["config.py", "segmentation.py", "features.py", "geometry.py", "still_policy.py",
             "generation.py", "action_language.py", "prompt_contract.py", "common/numeric.py",
             "common/model_client.py", "common/artifacts.py", "pretrain/abc_still.py",
             "pretrain/length_policy.py", "pretrain/segmentation.py", "pretrain/prompts.py",
             "pretrain/student_format.py", "pretrain/pipeline.py", "pretrain/realomni/reference.py",
             "pretrain/realomni/storage.py"]
    files = {name: file_hash(package / name) for name in names}
    files.update({str(p.relative_to(package)): file_hash(p) for p in Path(__file__).parent.iterdir()
                  if p.suffix in {".py", ".html"}})
    deps = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pyarrow", "av", "Pillow")}
    return {"files": files, "dependencies": deps}


def initialize(run, settings, selection=None):
    run, source = Path(run).resolve(), Path(settings.source_root).resolve()
    if not source.is_dir() or run == source or run in source.parents or source in run.parents:
        raise ValueError("existing source and output must be separate")
    value = {"schema": "agibot_whole_episode_16d_api_v2", "settings": settings.as_dict(),
             "selection": sorted(selection) if selection else None, "implementation": implementation(),
             "action_contract": action_contract(settings), "student_system": student_system(settings),
             "teacher_system": teacher_system(settings)}
    version = digest(value)
    value = json.loads(encode({"version": version, **value}))
    with lock(run / "control/initialize.lock", blocking=True):
        path = run / "run_config.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("code, prompt, dependencies or semantic configuration changed; use a new run directory")
        if not path.exists():
            atomic_json(path, value)
    inventory(run, settings, selection)
    return version


def episode_dir(run, row):
    bucket = f"shard_{row['shard']:04d}" if "shard" in row else row["key"][:2]
    return Path(run) / "episodes" / bucket / row["key"]


def stage_identity(version, row, name):
    return digest({"version": version, "source": row["source_identity"], "stage": name})


def save_arrays(path, arrays):
    np.savez_compressed(path, **arrays)


def load_numeric(directory):
    with np.load(directory / "native.npz", allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def episode_from(row, arrays, settings):
    stamps = np.rint(np.arange(row["length"])*1e9/settings.fps).astype(np.int64)
    e = build_reference(arrays, stamps, settings, episode_id=row["key"], task=row["task"],
                        source={"episode_id": row["episode_id"], "split": row["split"], "identity": row["source_identity"]})
    e["source_annotations"] = [{"start_frame": int(a["start_frame"]), "end_frame": int(a["end_frame"]),
                                "text": a["action_text"], "skill": a.get("skill")}
                               for a in row["metadata"].get("action_config", [])]
    return e


def publish(run, row, episode, segments, media, checked, version, settings):
    for segment in segments:
        validate_action_primitives(segment)
    target = Path(run) / "data" / row["split"] / row["key"][:2] / row["key"]
    identity = stage_identity(version, row, "export")
    if target.exists():
        verify_directory(target, identity)
        return target
    temporary = staging(target.parent, "export_" + row["key"])
    source_images = read_json(media / "images.json")
    cursor = 0
    metadata = []
    records = []
    for i, current in enumerate(segments):
        s, end = current["start_frame"], current["end_frame"]
        if s != cursor or not s <= end < len(episode["actions"]):
            raise ValueError("gapped or overlapping reference actions")
        cursor = end + 1
        target_value = target_for(episode, current)
        expected = np.asarray(episode["measured_states"][end+1])-episode["measured_states"][s]
        if len(target_value["guide_action"]) != 16 or not np.allclose(target_value["guide_action"], expected, atol=5e-8, rtol=0):
            raise ValueError("16D guide mismatch")
        user, frames, roles = build_user(episode, current, segments[i-1] if i else None, settings.student_cameras)
        images = []
        image_meta = []
        for when, frame in frames:
            for camera in settings.student_cameras:
                relative = source_images[camera][str(frame)]
                p = media / relative
                if not p.is_file():
                    raise ValueError("missing persistent student image")
                images.append(str(p.resolve()))
                image_meta.append({"camera": camera, "observation_frame": frame, "role": when,
                                   "video_timestamp_s": row["videos"][camera]["from_s"] + frame/settings.fps})
        records.append({"messages": [{"role": "system", "content": student_system(settings)},
                                      {"role": "user", "content": user},
                                      {"role": "assistant", "content": json.dumps(target_value, ensure_ascii=False)}], "images": images})
        metadata.append({"row_id": f"{row['episode_id']}/segment_{i:05d}", "episode_id": row["episode_id"],
                         "segment": current, "images": image_meta, "source_identity": row["source_identity"],
                         "action_rows": [s, end], "target_observation_frame": end+1})
    if cursor != row["length"]-1:
        raise ValueError("export omitted real state transitions")
    atomic_bytes(temporary / "vlm.jsonl", ("".join(encode(r)+"\n" for r in records)).encode())
    atomic_bytes(temporary / "metadata.jsonl", ("".join(encode(r)+"\n" for r in metadata)).encode())
    measured = np.asarray(episode["measured_states"])
    save_arrays(temporary / "reference.npz", {"measured_states": measured, "states": measured[:-1],
                "reference_actions": measured[1:], "timestamps_ns": np.asarray(episode["measured_timestamps_ns"])})
    atomic_json(temporary / "vla.json", {"episode_id": row["episode_id"], "task": row["task"],
        "source": episode["source"], "contract": action_contract(settings), "arrays": "reference.npz",
        "native_arrays": str((checked / "native.npz").resolve()), "segments": segments})
    atomic_json(temporary / "VALIDATION.json", {"whole_episode": True, "source_observations": row["length"],
        "reference_actions": cursor, "segments": len(segments), "dimension": 16, "coverage": "all N-1 transitions",
        "guide_verified": True, "causal_student_images": True, "no_padding_actions": True, "noncot": True})
    commit_directory(temporary, target, identity)
    return target


def process_episode(run, row, settings, version, reader, *, endpoints=(), stage="all", fault=None, api_key=None):
    directory = episode_dir(run, row)
    with lock(directory / "episode.lock"):
        return _process(run, row, settings, version, reader, endpoints, stage, fault, api_key)


def _process(run, row, settings, version, reader, endpoints, stage, fault, api_key):
    directory = episode_dir(run, row)
    status_path = directory / "status.json"
    old = read_json(status_path) if status_path.exists() else {}
    if old and (old["version"] != version or old["source_identity"] != row["source_identity"]):
        raise RuntimeError("episode checkpoint identity changed")
    record = {**old, "episode_id": row["episode_id"], "key": row["key"],
              "version": version, "source_identity": row["source_identity"],
              "owner": process_identity(), "attempts": old.get("attempts", 0)+1}
    journal = Journal(directory / "events.jsonl")

    def update(state, **details):
        record.update(status=state, updated_at=time.time(), **details)
        atomic_json(status_path, record)
        journal.append({"time": record["updated_at"], "status": state, **details})
        if fault:
            fault(state)
        return record

    def committed(name):
        path = directory / name
        if path.exists():
            verify_directory(path, stage_identity(version, row, name))
            return True
        return False

    def finish(temporary, name):
        check_sources(row)
        commit_directory(temporary, directory / name, stage_identity(version, row, name))
        if fault:
            fault("committed_" + name)

    try:
        check_sources(row)
        if old.get("status") == "rejected":
            verify_directory(directory / "rejected", stage_identity(version, row, "rejected"))
            return old
        if old.get("status") == "complete":
            verify_directory(Path(old["export_path"]), stage_identity(version, row, "export"))
            verify_directory(directory / "media", stage_identity(version, row, "media"))
            verify_directory(directory / "numeric", stage_identity(version, row, "numeric"))
            return old
        for p in directory.glob(".pending_*"):
            shutil.rmtree(p)

        def reject(audit):
            if not committed("rejected"):
                temporary = staging(directory, "rejected")
                atomic_json(temporary / "decision.json", audit)
                finish(temporary, "rejected")
            return update("rejected", checks=audit["checks"], reasons=audit["reasons"], retryable=False)

        if committed("rejected"):
            return reject(read_json(directory / "rejected/decision.json"))
        if row.get("catalog_errors"):
            return reject({"checks": {"numeric": "failed", "base": "not_checked", "video": "not_checked"},
                "reasons": [{"code": "task_parquet_catalog_unverifiable", "evidence": row["catalog_errors"]}],
                "whole_episode": True, "trimmed_rows": 0})
        if not committed("numeric"):
            update("checking_numeric", retryable=True)
            arrays = reader.read(row)
            audit = inspect_numeric(row, arrays, settings)
            if audit["reasons"]:
                return reject(audit)
            temporary = staging(directory, "numeric")
            save_arrays(temporary / "native.npz", arrays)
            atomic_json(temporary / "audit.json", audit)
            finish(temporary, "numeric")
        arrays = load_numeric(directory / "numeric")
        audit = read_json(directory / "numeric/audit.json")
        update("numeric_passed", checks=audit["checks"])
        for camera in settings.cameras:
            name = "camera_" + camera
            if committed(name):
                continue
            update("checking_video", current_camera=camera)
            try:
                result = read_video(row["videos"][camera], row["length"], settings.fps, settings.video_tolerance_s,
                                    heartbeat=lambda count: update("checking_video", current_camera=camera,
                                                                   video_frames_checked=count))
            except (InvalidVideo, av_error()) as error:
                audit["checks"]["video"] = "failed"
                audit["reasons"].append({"code": "incomplete_or_invalid_video", "camera": camera, "detail": str(error)})
                return reject(audit)
            temporary = staging(directory, name)
            atomic_json(temporary / "audit.json", result)
            finish(temporary, name)
        audit["checks"]["video"] = "passed"
        if not committed("checked"):
            temporary = staging(directory, "checked")
            atomic_json(temporary / "decision.json", {**audit, "decision": "accepted"})
            finish(temporary, "checked")
        update("checked", checks=audit["checks"], retryable=False)
        if stage == "inspect":
            return record
        episode = episode_from(row, arrays, settings)
        if not committed("segmented"):
            update("segmenting")
            result = segment(episode, settings)
            temporary = staging(directory, "segmented")
            atomic_json(temporary / "segmentation.json", result)
            finish(temporary, "segmented")
        result = read_json(directory / "segmented/segmentation.json")
        update("segmented", segments=len(result["segments"]), reference_action_rows=row["length"]-1)
        if stage == "segment":
            return record
        if not committed("media"):
            update("materializing")
            temporary = staging(directory, "media")
            indices = sample_frames(result["segments"])
            images = {}
            for camera in settings.student_cameras:
                selected = read_video(row["videos"][camera], row["length"], settings.fps, settings.video_tolerance_s,
                                      selected=indices, destination=temporary / camera)
                images[camera] = {str(frame): str(Path(p).relative_to(temporary)) for frame, p in selected["paths"].items()}
            (temporary / "teacher").mkdir()
            for frame in indices:
                montage([temporary / images[c][str(frame)] for c in settings.student_cameras],
                        settings.student_cameras, temporary / "teacher" / f"{frame:07d}.jpg")
            atomic_json(temporary / "images.json", images)
            finish(temporary, "media")
        update("media_ready")
        if stage == "media":
            return record
        if not committed("annotated"):
            update("annotating", retryable=True)
            labels = annotate(episode, result, directory / "media", directory / "labels", settings, endpoints,
                              lambda done, total: update("annotating", segments_done=done, segments_total=total),
                              api_key=api_key)
            temporary = staging(directory, "annotated")
            atomic_json(temporary / "segments.json", labels)
            finish(temporary, "annotated")
        segments = read_json(directory / "annotated/segments.json")
        check_sources(row)
        update("exporting")
        output = publish(run, row, episode, segments, directory / "media", directory / "numeric", version, settings)
        if fault:
            fault("committed_export")
        return update("complete", export_path=str(output.resolve()), segments_done=len(segments), retryable=False, error=None)
    except AnnotationRejected as error:
        return update("annotation_rejected", retryable=False, error=str(error))
    except (ServiceUnavailable, OSError) as error:
        return update("retryable_error", retryable=True, error=f"{type(error).__name__}: {error}")
    except Exception as error:
        update("processing_error", retryable=False, error=f"{type(error).__name__}: {error}")
        raise


def av_error():
    import av
    return av.error.InvalidDataError


def work(run, settings, endpoints, stage="all", retry_passes=3, *, concurrency=4, api_key=None):
    run = Path(run)
    if concurrency < 1 or retry_passes < 1:
        raise ValueError("concurrency and retry passes must be positive")
    if not (run / "inventory/COMMIT.json").is_file():
        raise RuntimeError("source inventory is not committed; initialize must finish before workers start")
    config = read_json(run / "run_config.json")
    expected = json.loads(encode(settings.as_dict()))
    if config["settings"] != expected or config["implementation"] != implementation():
        raise RuntimeError("worker did not import the frozen run implementation/dependencies")
    version = config["version"]
    if stage == "all" and (not endpoints or not api_key):
        raise ServiceUnavailable("API endpoint and token are required for annotation")
    identity = digest({"source": str(Path(settings.source_root).resolve()), "settings": expected,
                       "selection": config["selection"]})
    verify_directory(run / "inventory", identity)
    local = threading.local()

    def process(row):
        if not hasattr(local, "reader"):
            local.reader = NumericReader()
        try:
            return process_episode(run, row, settings, version, local.reader,
                                   endpoints=endpoints, stage=stage, api_key=api_key)
        except Exception as error:
            path = episode_dir(run, row) / "status.json"
            previous = read_json(path) if path.exists() else {}
            return {**previous, "status": "processing_error", "error": type(error).__name__}

    with lock(run / "control/worker.lock"):
        for attempt in range(retry_passes):
            counts = Counter()
            started = time.time()
            rows = iter(iter_inventory(run))
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending = {}

                def submit():
                    row = next(rows, None)
                    if row is not None:
                        pending[executor.submit(process, row)] = row

                for _ in range(concurrency):
                    submit()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = pending.pop(future)
                        record = future.result()
                        counts[record["status"]] += 1
                        atomic_json(run / "worker_progress.json", {
                            "concurrency": concurrency, "owner": process_identity(),
                            "updated_at": time.time(), "started_at": started,
                            "counts": dict(counts), "pass": attempt, "stage": stage})
                        print(encode({"episode": row["episode_id"], "status": record["status"]}), flush=True)
                        submit()
            summarize(run)
            if not counts["retryable_error"] and not counts["processing_error"]:
                return dict(counts)
            if counts["processing_error"] and not counts["retryable_error"]:
                raise RuntimeError("processing errors remain; inspect progress.json and episode status files")
            if attempt+1 < retry_passes:
                time.sleep(min(30, 5*2**attempt))
        raise RuntimeError("retryable episodes remain; rerun the same script to continue")


def summarize(run):
    run = Path(run)
    if not (run / "inventory/COMMIT.json").is_file():
        raise RuntimeError("inventory has not been committed")
    counts = Counter()
    reasons = Counter()
    report_path = run / "progress.json"
    with lock(report_path.with_suffix(".lock"), blocking=True):
        manifest = run / "accepted.jsonl"
        with manifest.with_suffix(".tmp").open("w") as stream:
            for row in iter_index(run):
                path = episode_dir(run, row) / "status.json"
                status = read_json(path) if path.exists() else {"status": "pending"}
                counts[status["status"]] += 1
                reasons.update(r["code"] for r in status.get("reasons", []))
                if status["status"] == "complete":
                    commit = read_json(Path(status["export_path"]) / "COMMIT.json")
                    expected = stage_identity(status["version"], row, "export")
                    if commit["identity"] != expected:
                        raise RuntimeError("accepted episode export identity mismatch")
                    stream.write(encode({"episode_id": row["episode_id"], "split": row["split"],
                        "path": status["export_path"], "segments": status["segments_done"]})+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest.with_suffix(".tmp"), manifest)
        result = {"updated_at": time.time(), "counts": dict(counts), "rejection_reasons": dict(reasons),
                  "scope": "all enumerated episodes", "accepted_is_whole_episode": True}
        atomic_json(report_path, result)
    return result
