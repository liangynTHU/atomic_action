#!/usr/bin/env python3
"""Build a portable review from committed AgiBot exports, without running workers."""

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_seg.pretrain.agibot.media import read_video
from atomic_seg.pretrain.agibot.source import iter_inventory
from atomic_seg.pretrain.agibot.runner import episode_dir
from atomic_seg.pretrain.realomni.storage import file_hash, read_json, verify_directory


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def preview_video(job):
    target = Path(job["target"])
    report = target.with_suffix(".json")
    identity = {k: job[k] for k in ("source", "frames", "fps", "tolerance")}
    if target.is_file() and report.is_file():
        cached = read_json(report)
        if cached.get("identity") == identity and cached.get("sha256") == file_hash(target):
            return job["episode_id"], job["camera"], cached
    if job.get("reuse"):
        reusable = Path(job["reuse"])
        if reusable.is_file() and reusable.with_suffix(".json").is_file():
            cached = read_json(reusable.with_suffix(".json"))
            if cached.get("identity") == identity and cached.get("sha256") == file_hash(reusable):
                shutil.copyfile(reusable, target)
                write_json(report, cached)
                return job["episode_id"], job["camera"], cached
    temporary = target.with_name(target.stem + ".pending.mp4")
    result = read_video(job["source"], job["frames"], job["fps"], job["tolerance"], preview=temporary)
    temporary.replace(target)
    result = {"identity": identity, "validation": result, "sha256": file_hash(target)}
    write_json(report, result)
    return job["episode_id"], job["camera"], result


def build(run, output, episode_ids, limit, workers, reuse_videos_from=None):
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents:
        raise ValueError("Keep the review outside the production run directory")
    config = read_json(run / "run_config.json")
    settings = config["settings"]
    fps = settings["fps"]
    candidates = {}
    manifests = [run / "accepted.jsonl"]
    if not manifests[0].is_file():
        manifests = sorted((run / "groups").glob("*/accepted.jsonl"))
    for manifest in manifests:
        if not manifest.exists():
            continue
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            candidates.setdefault(entry["episode_id"], {**entry, "manifest": str(manifest)})
    if episode_ids:
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("Duplicate selected episode")
        selected = [candidates[key] for key in episode_ids]
    else:
        selected, tasks = [], set()
        for entry in candidates.values():
            task = entry["episode_id"].split("/")[0]
            if task not in tasks:
                selected.append(entry)
                tasks.add(task)
            if len(selected) == limit:
                break
    if not selected:
        raise ValueError("No committed episodes selected")
    inventory = {}
    keys = {Path(entry["path"]).name for entry in selected}
    for row in iter_inventory(run):
        if row["key"] in keys:
            inventory[row["key"]] = row
            if len(inventory) == len(keys):
                break
    output.mkdir(parents=True, exist_ok=True)
    examples, jobs, provenance = [], [], []
    for entry in selected:
        export = Path(entry["path"])
        row = inventory[export.name]
        directory = episode_dir(run, row)
        status = read_json(directory / "status.json")
        if status["status"] != "complete":
            raise ValueError(f"Episode is not complete: {entry['episode_id']}")
        commit = read_json(export / "COMMIT.json")
        verify_directory(export, commit["identity"])
        vla = read_json(export / "vla.json")
        validation = read_json(export / "VALIDATION.json")
        vlm = [json.loads(line) for line in (export / "vlm.jsonl").read_text().splitlines()]
        metadata = [json.loads(line) for line in (export / "metadata.jsonl").read_text().splitlines()]
        with np.load(export / "reference.npz", allow_pickle=False) as arrays:
            measured = arrays["measured_states"].copy()
        assert measured.shape == (row["length"], 16)
        assert len(vla["segments"]) == len(vlm) == len(metadata) == entry["segments"]
        target = output / export.name[:16]
        target.mkdir(exist_ok=True)
        original = target / "export"
        original.mkdir(exist_ok=True)
        for name in [x["path"] for x in commit["files"]] + ["COMMIT.json"]:
            shutil.copyfile(export / name, original / name)
        shutil.copyfile(directory / "checked/decision.json", target / "decision.json")
        shutil.copyfile(directory / "segmented/segmentation.json", target / "segmentation.json")
        downloads = {}
        for name in ("vlm.jsonl", "vla.json", "reference.npz", "VALIDATION.json", "decision.json"):
            path = target / name if name == "decision.json" else original / name
            downloads[name] = {"base64": base64.b64encode(path.read_bytes()).decode("ascii")}
        segments, cursor = [], 0
        for index, (segment, training, meta) in enumerate(zip(vla["segments"], vlm, metadata)):
            start, end = segment["start_frame"], segment["end_frame"] + 1
            assert start == cursor and end < len(measured)
            cursor = end
            assert np.allclose(segment["guide_action"], measured[end] - measured[start], atol=5e-8, rtol=0)
            assistant = json.loads(next(m["content"] for m in training["messages"] if m["role"] == "assistant"))
            assert all(assistant[k] == segment[k] for k in ("guide_action", "primary_action_verb", "sub_task", "num_chunks"))
            shots = []
            for frame in (start, (start + end) // 2, end):
                destination = target / f"observation_{frame:07d}.jpg"
                shutil.copyfile(directory / "media/teacher" / f"{frame:07d}.jpg", destination)
                shots.append({"frame": frame, "path": destination.relative_to(output).as_posix()})
            student_images = []
            assert len(training["images"]) == len(meta["images"])
            for source, image_meta in zip(training["images"], meta["images"]):
                destination = target / "student" / image_meta["camera"] / Path(source).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(source, destination)
                student_images.append({**image_meta, "path": destination.relative_to(output).as_posix()})
            segments.append({**segment, "shots": shots, "start_s": start / fps, "end_s": end / fps,
                             "messages": training["messages"], "student_images": student_images})
        assert cursor == row["length"] - 1
        sample = np.unique(np.linspace(0, len(measured) - 1, min(900, len(measured)), dtype=int))
        cameras = []
        for camera in settings["student_cameras"]:
            video = target / f"{camera}.mp4"
            cameras.append({"name": camera, "video": video.relative_to(output).as_posix()})
            jobs.append({"episode_id": row["episode_id"], "camera": camera, "source": row["videos"][camera],
                         "frames": row["length"], "fps": fps, "tolerance": settings["video_tolerance_s"],
                         "target": str(video),
                         "reuse": str(reuse_videos_from / export.name[:16] / video.name)
                         if reuse_videos_from else None})
        examples.append({"id": export.name[:16], "episode_id": row["episode_id"], "task": row["task"],
                         "status": "complete", "source_rows": row["length"],
                         "reference_rows": len(measured) - 1, "segments": segments, "cameras": cameras,
                         "model_labels": True, "checks": status.get("checks", {}), "validation": validation,
                         "downloads": downloads,
                         "source_export": str(export), "duration_s": row["length"] / fps,
                         "signal": {"frames": sample.tolist(), "left_gripper": measured[sample, 7].tolist(),
                                    "right_gripper": measured[sample, 15].tolist(),
                                    "left_xyz": measured[sample, :3].tolist(), "right_xyz": measured[sample, 8:11].tolist()}})
        provenance.append({**entry, "source_identity": row["source_identity"], "source_export_commit": commit})
        print("Prepared", row["episode_id"], len(segments), "segments", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(preview_video, job) for job in jobs]
        for future in as_completed(futures):
            episode_id, camera, result = future.result()
            print("Video ready", episode_id, camera, result["validation"]["frames"], "frames", flush=True)
    data = {"run": str(run), "source": settings["source_root"], "examples": examples, "fps": fps,
            "contract": config["action_contract"], "student_system": config["student_system"],
            "teacher_system": config["teacher_system"], "version": config["version"],
            "created_at": datetime.now(timezone.utc).isoformat(), "selection": provenance}
    write_json(output / "examples.json", data)
    write_json(output / "selection.json", [x["episode_id"] for x in examples])
    write_json(output / "DATA_VALIDATION.json", {"passed": True, "committed_exports_verified": True,
               "exact_existing_labels_and_messages": True, "guide_verified": True,
               "episodes": len(examples), "segments": sum(len(e["segments"]) for e in examples),
               "reference_actions": sum(e["reference_rows"] for e in examples), "videos": len(jobs),
               "source_run": str(run), "selection": provenance})
    template = Path(__file__).with_name("agibot_completed_review.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__DATA__", payload))
    shutil.copyfile(Path(__file__).parent / "assets/agibot-completed-review-cjk.woff", output / "review.woff")
    print(output / "index.html", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-videos-from", type=Path,
                        help="Reuse preview videos only when source identities and SHA256 match")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        parser.error("limit and workers must be positive")
    if args.selection and args.episode:
        parser.error("Use either --selection or --episode")
    episode_ids = read_json(args.selection) if args.selection else args.episode
    build(args.run, args.output, episode_ids, args.limit, args.workers, args.reuse_videos_from)


if __name__ == "__main__":
    main()
