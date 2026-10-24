#!/usr/bin/env python3
"""Review real cropped RealOmni trajectories, atomic segments and 16D targets."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import av
import numpy as np
from PIL import Image

from build_realomni_review import extract_media, save_json, title_for
from atomic_seg.pretrain.realomni.runner import load_episode
from atomic_seg.pretrain.realomni.storage import (
    atomic_bytes, atomic_json, check_source, digest, episode_dir, file_hash,
    iter_jsonl, read_json, verify_directory,
)


def reusable_media(root, episode_id, sha256, destination, name):
    if root is None or not (root / "examples.json").exists():
        return None
    data = read_json(root / "examples.json")
    previous = next((e for e in data["examples"] if e["episode_id"] == episode_id and e["source_sha256"] == sha256
                     and len(e.get("cameras", [])) == 2 and not e.get("media_errors")), None)
    if previous is None:
        return None
    cameras = copy.deepcopy(previous["cameras"])
    for camera in cameras:
        source = root / camera["video"]
        with av.open(str(source)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            count = sum(1 for _ in container.decode(video=0))
        if count != camera["expected_frames"]:
            raise ValueError(f"cached source preview is incomplete: {source}")
        for key in ("video", "poster"):
            path = root / camera[key]
            shutil.copyfile(path, destination / path.name)
            camera[key] = f"{name}/{path.name}"
        camera["storyboard"] = []
    return cameras


def segment_frames(root, destination, cameras, segments, stamps, origin):
    samples = sorted({int(i) for s in segments for i in (s["start_frame"], (s["start_frame"] + s["end_frame"] + 1) // 2,
                                                        s["end_frame"] + 1)})
    shots = {frame: {} for frame in samples}
    for camera in cameras:
        source_times = [(int(stamps[frame]) - camera["first_timestamp_ns"]) / 1e9 for frame in samples]
        last_image, last_time, next_sample = None, None, 0

        def save(frame, im, video_time):
            path = destination / f"{camera['name']}_observation_{frame:06d}.jpg"
            im.save(path, quality=86)
            shots[frame][camera["name"]] = {"path": str(path.relative_to(root)), "source_time_s": (int(stamps[frame]) - origin) / 1e9,
                                           "video_frame_time_s": video_time, "observation_frame": frame}

        with av.open(str(root / camera["video"])) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for decoded in container.decode(video=0):
                t = float(decoded.time)
                while next_sample < len(samples) and source_times[next_sample] < t - 1 / 90_000:
                    if last_image is None:
                        raise ValueError("no causal preview frame at retained observation")
                    save(samples[next_sample], last_image, last_time)
                    next_sample += 1
                last_image, last_time = decoded.to_image(), t
                if next_sample == len(samples):
                    break
            while next_sample < len(samples):
                if last_image is None:
                    raise ValueError("empty source preview")
                save(samples[next_sample], last_image, last_time)
                next_sample += 1
    for s in segments:
        frames = [s["start_frame"], (s["start_frame"] + s["end_frame"] + 1) // 2, s["end_frame"] + 1]
        s["review_frames"] = [{"frame": frame, "source_time_s": (int(stamps[frame]) - origin) / 1e9,
                               "images": shots[frame]} for frame in frames]


def write_page(output, data):
    names = ["examples.json", "all_results.jsonl"]
    for e in data["examples"]:
        names.extend([f"{e['id']}/decision.json", f"{e['id']}/segmentation.json", f"{e['id']}/reference_actions.json"])
    downloads = {name: (output / name).read_text() for name in names}
    template = Path(__file__).with_name("realomni_trim_review.html").read_text()
    text = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
    text = text.replace("__DOWNLOADS__", json.dumps(downloads, ensure_ascii=False).replace("<", "\\u003c"))
    atomic_bytes(output / "index.html", text.encode())


def build(run, output, *, previous=None, limit=6):
    run, output = run.resolve(), output.resolve()
    config = read_json(run / "run_config.json")
    if config["settings"].get("quality_policy") != "trim_edges_1s":
        raise ValueError("this review requires the one-second edge crop policy")
    if output.exists() and any(output.iterdir()):
        raise ValueError("select a new preview output directory")
    source_root = Path(config["settings"]["source_root"]).resolve()
    if output == run or output in run.parents or output == source_root or source_root in output.parents:
        raise ValueError("preview output must be separate from source data")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in iter_jsonl(run / "inventory/episodes.jsonl"):
        directory = episode_dir(run, row)
        status = read_json(directory / "status.json")
        verify_directory(directory / "checked", digest({"version": config["version"], "source": row["source_identity"], "stage": "checked"}))
        decision = read_json(directory / "checked/decision.json")
        audit = decision["audit"]
        records.append({"episode_id": row["episode_id"], "source_file": row["file"]["path"],
                        "source_sha256": decision["source_sha256"], "title": title_for(row["episode_id"]),
                        "status": status["status"], "segments": status.get("segments", 0),
                        "reference_action_rows": status.get("reference_action_rows", 0),
                        "crop": audit.get("crop"), "reasons": audit["reasons"],
                        "diagnostics": audit.get("diagnostics", []), "key": row["key"], "row": row})
    candidates = [r for r in records if r["status"] == "segmented"]
    priority = ["clean_bowl", "desktop_object_sorting", "lace_up_shoes_with_both_hands", "fold_and_store_clothes",
                "clean_container", "unscrew_bottle_cap_and_pour"]
    selected = []
    for task in priority:
        match = next((r for r in candidates if task in Path(r["episode_id"]).parts and r not in selected), None)
        if match:
            selected.append(match)
    selected.extend(r for r in candidates if r not in selected)
    examples = []
    for record in selected[:limit]:
        name = f"example_{len(examples) + 1:02d}"
        target = output / name
        target.mkdir()
        row = record["row"]
        check_source(row)
        if file_hash(record["source_file"]) != record["source_sha256"]:
            raise ValueError("source content changed")
        directory = episode_dir(run, row)
        verify_directory(directory / "segmented", digest({"version": config["version"], "source": row["source_identity"], "stage": "segmented"}))
        result = read_json(directory / "segmented/segmentation.json")
        episode = load_episode(directory / "checked")
        decision = read_json(directory / "checked/decision.json")
        save_json(target / "decision.json", decision)
        save_json(target / "segmentation.json", result)
        shutil.copyfile(directory / "checked/trajectory.npz", target / "trajectory.npz")
        crop = episode["crop"]
        origin = crop["original_start_ns"]
        print("review", record["episode_id"], len(result["segments"]), "segments", flush=True)
        cameras = reusable_media(previous, record["episode_id"], record["source_sha256"], target, name)
        if cameras is None:
            camera_dict, _, _, errors = extract_media(record["source_file"], target, decision["audit"])
            if errors:
                raise ValueError(f"source preview failed: {errors}")
            cameras = list(camera_dict.values())
            for camera in cameras:
                for key in ("video", "poster"):
                    camera[key] = f"{name}/{camera[key]}"
                camera["storyboard"] = []
        for camera in cameras:
            camera["offset_s"] = (camera["first_timestamp_ns"] - origin) / 1e9
            camera["end_s"] = (camera["last_timestamp_ns"] - origin) / 1e9
        segments = copy.deepcopy(result["segments"])
        measured = np.asarray(episode["measured_states"])
        stamps = episode["measured_timestamps_ns"]
        for s in segments:
            start, end = s["start_frame"], s["end_frame"]
            expected = np.round(measured[end + 1] - measured[start], 8).tolist()
            if s["guide_action"] != expected or s["num_chunks"] != (end - start + 8) // 8:
                raise ValueError("guide or chunk arithmetic failed")
            s.update(state_at_start=measured[start].tolist(), target_state=measured[end + 1].tolist(),
                     first_reference_action=measured[start + 1].tolist(),
                     padding_rows_if_chunked=8 * s["num_chunks"] - s["valid_action_rows"])
        segment_frames(output, target, cameras, segments, stamps, origin)
        times = [(int(t) - origin) / 1e9 for t in stamps]
        save_json(target / "reference_actions.json", {"contract": result["action_contract"], "crop": crop,
                  "measured_states": measured.tolist(), "measured_timestamps_ns": stamps,
                  "reference_actions": measured[1:].tolist(), "states": measured[:-1].tolist()})
        plot_stride = max(1, len(measured) // 600)
        indices = sorted(set(range(0, len(measured), plot_stride)) | {len(measured) - 1})
        example = {k: v for k, v in record.items() if k != "row"}
        example.update(id=name, cameras=cameras, segments=segments, duration_s=(crop["original_end_ns"] - origin) / 1e9,
                       retained_start_s=times[0], retained_end_s=times[-1],
                       measured_observations=len(measured), guide_dimension=16,
                       plot={"times": [times[i] for i in indices], "left_gripper": measured[indices, 7].tolist(),
                             "right_gripper": measured[indices, 15].tolist()},
                       still_audit=result["both_still_cleanup"], length_audit=result["length_audit"],
                       whole_original_episode_retained=False, semantic_annotation="not_run; numerical motion labels only")
        save_json(target / "example.json", example)
        examples.append(example)
        check_source(row)
    for r in records:
        r.pop("row")
    data = {"schema": "realomni_trim_review_v1", "summary": dict(Counter(r["status"] for r in records)),
            "source_episodes": len(records), "total_segments": sum(r["segments"] for r in records),
            "total_action_rows": sum(r["reference_action_rows"] for r in records),
            "examples": examples, "all_results": records,
            "policy": {"max_trim_per_edge_seconds": 1, "imu": "diagnostic_only", "calibration": "diagnostic_only",
                       "interior_required_gaps": "reject_episode", "base_filter": False},
            "previous_results": {"source_episodes": len(records), "segmented": 0}, "run_root": str(run)}
    save_json(output / "examples.json", data)
    atomic_bytes(output / "all_results.jsonl", ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)).encode())
    write_page(output, data)
    shutil.copyfile(Path(__file__).parent / "assets/realomni-trim-review-cjk.woff", output / "review-cjk.woff")
    shutil.copyfile(Path(__file__).parent / "assets/OFL.txt", output / "OFL.txt")
    save_json(output / "VALIDATION.json", {"source_episodes": len(records), "examples": len(examples),
              "all_guide_actions_verified": True, "all_chunks_verified": True, "source_identity_checked": True,
              "full_retained_transition_coverage": True, "semantics": "numerical labels only"})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-media-from", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    result = build(args.run_root, args.output_root, previous=args.reuse_media_from, limit=args.limit)
    print(json.dumps({"index": str(args.output_root / "index.html"), "summary": result["summary"],
                      "segments": result["total_segments"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
