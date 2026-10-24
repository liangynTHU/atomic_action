#!/usr/bin/env python3
"""Build a portable review of committed ABC/AgiBot and current RoboTwin data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_episode_dataset_review import (
    DISPLAY_FPS, camera_video_frames, composite_frames, encode_video,
)

FIELDS = ("guide_action", "primary_action_verb", "sub_task", "num_chunks")
ROBOTWIN_IDS = (525, 1075, 4925, 25825)
ABC_TASKS = (
    "fold and stack the shorts", "erase the whiteboard",
    "clip the underwear to the hanger", "build the wood block tower",
)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def copy_asset(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != sha(destination):
        raise ValueError(f"Copied asset differs: {source}")


def verify_rows(segments, rows, states, actions, expected_rows, tolerance):
    if len(segments) != len(rows):
        raise ValueError("Missing segment training rows")
    cursor, maximum_error = 0, 0.0
    for segment, row in zip(segments, rows):
        start, end = segment["start_frame"], segment["end_frame"]
        if start != cursor or end < start:
            raise ValueError("Gapped or overlapping segmentation")
        if set(row) != {"messages", "images"}:
            raise ValueError("Unexpected training row keys")
        if [message["role"] for message in row["messages"]] != ["system", "user", "assistant"]:
            raise ValueError("Unexpected training message roles")
        target = json.loads(row["messages"][-1]["content"])
        if set(target) != set(FIELDS):
            raise ValueError("Unexpected non-CoT answer keys")
        if any(target[key] != segment[key] for key in FIELDS):
            raise ValueError("Training row and segment differ")
        if target["num_chunks"] != math.ceil((end - start + 1) / 8):
            raise ValueError("Invalid inclusive action chunks")
        error = float(np.max(np.abs(actions[end] - states[start] - target["guide_action"])))
        if error > tolerance:
            raise ValueError(f"Guide differs from source trajectory: {error}")
        maximum_error = max(maximum_error, error)
        if row["messages"][1]["content"].count("<image>") != len(row["images"]):
            raise ValueError("Image tokens and paths differ")
        if any(not Path(path).is_file() for path in row["images"]):
            raise ValueError("Missing original student observation")
        cursor = end + 1
    if cursor != expected_rows:
        raise ValueError("Incomplete episode action coverage")
    return {"passed": True, "action_rows": cursor, "segments": len(rows),
            "guide_max_error": maximum_error, "guide_tolerance": tolerance,
            "complete_coverage": True, "noncot_four_field_targets": True,
            "inclusive_chunks_verified": True}


def build_abc(source, output):
    data = read(source / "complete_episodes.json")
    selected = [next(e for e in data["episodes"] if e["task"] == task) for task in ABC_TASKS]
    result = []
    for index, original in enumerate(selected, 1):
        episode = copy.deepcopy(original)
        source_dir = source / original["directory"]
        destination = output / "episodes" / f"abc_{index:02d}"
        commit_path = Path(original["production_commit"])
        commit = read(commit_path)
        if sha(commit_path) != original["production_commit_sha256"]:
            raise ValueError("ABC source commit changed")
        files = {item["path"]: item["sha256"] for item in commit["files"]}
        split = original["source"]["split"]
        rows_path = commit_path.parent / f"data/vlm/{split}/accepted.jsonl"
        metadata_path = commit_path.parent / f"metadata/{split}/accepted.jsonl"
        for path in (rows_path, metadata_path):
            if sha(path) != files[path.relative_to(commit_path.parent).as_posix()]:
                raise ValueError("ABC committed file hash changed")
        raw_lines, metadata_lines = [], []
        with rows_path.open() as rows_file, metadata_path.open() as metadata_file:
            for raw, meta_raw in zip_longest(rows_file, metadata_file):
                if raw is None or meta_raw is None:
                    raise ValueError("ABC training/metadata line counts differ")
                if json.loads(meta_raw)["episode_id"] == original["id"]:
                    raw_lines.append(raw)
                    metadata_lines.append(meta_raw)
        if "".join(raw_lines) != original["vlm_jsonl_text"]:
            raise ValueError("ABC cached preview differs from committed training rows")
        if [json.loads(line) for line in metadata_lines] != original["vlm_metadata"]:
            raise ValueError("ABC cached metadata differs from source")
        native = original["native_action_ref"]
        if sha(native["path"]) != native["sha256"]:
            raise ValueError("ABC native action hash changed")
        with np.load(native["path"], allow_pickle=False) as arrays:
            check = verify_rows(original["segments"], original["vlm"], arrays["states"],
                                arrays["actions"], original["frame_count"], 5e-8)
        shutil.copytree(source_dir, destination)
        for name in ("vlm.jsonl", "vlm_metadata.jsonl"):
            expected = "".join(raw_lines if name == "vlm.jsonl" else metadata_lines)
            if (destination / name).read_text() != expected:
                raise ValueError("ABC copied training rows changed")
        copy_asset(native["path"], destination / "native.npz")
        old_prefix, new_prefix = original["directory"] + "/", destination.relative_to(output).as_posix() + "/"
        episode["observations"] = [[path.replace(old_prefix, new_prefix, 1) for path in paths]
                                   for paths in episode["observations"]]
        for story in episode["storyboard"]:
            for shot in story:
                shot["path"] = shot["path"].replace(old_prefix, new_prefix, 1)
        episode.update(id=f"abc_{index:02d}", source_episode_id=original["id"],
                       directory=destination.relative_to(output).as_posix(),
                       video=original["video"].replace(old_prefix, new_prefix, 1),
                       poster=original["poster"].replace(old_prefix, new_prefix, 1),
                       display_timestamps=original["timestamps"], version="softchunks · 2026-10-13",
                       checks={**check, "committed_rows_byte_identical": True,
                               "commit_sha256": sha(commit_path), "native_sha256": native["sha256"]})
        episode["videos"] = [{"name": "三视角", "path": episode["video"]}]
        episode["downloads"] = ["vlm.jsonl", "vla.json", "vlm_metadata.jsonl",
                                "trajectory.json", "segmentation.json", "native.npz"]
        dump(destination / "VALIDATION.json", episode["checks"])
        result.append(episode)
        print("ABC ready", original["id"], len(episode["segments"]), flush=True)
    return result


def build_agibot(source, output):
    data = read(source / "examples.json")
    if len(data["examples"]) != 4:
        raise ValueError("Expected four previously reviewed committed AgiBot episodes")
    result = []
    for index, original in enumerate(data["examples"], 1):
        export = Path(original["source_export"])
        commit = read(export / "COMMIT.json")
        for item in commit["files"]:
            if sha(export / item["path"]) != item["sha256"]:
                raise ValueError("AgiBot committed export changed")
        raw = (export / "vlm.jsonl").read_text()
        rows = [json.loads(line) for line in raw.splitlines()]
        metadata = list(jsonl(export / "metadata.jsonl"))
        vla = read(export / "vla.json")
        segments = copy.deepcopy(vla["segments"])
        if len(rows) != len(metadata) or len(rows) != len(original["segments"]):
            raise ValueError("AgiBot cached training row count differs from source")
        if rows != [{"messages": s["messages"], "images": row["images"]}
                    for s, row in zip(original["segments"], rows)]:
            raise ValueError("AgiBot previous preview differs from source")
        destination = output / "episodes" / f"agibot_{index:02d}"
        shutil.copytree(source / original["id"], destination)
        copy_asset(export / "vlm.jsonl", destination / "vlm.jsonl")
        copy_asset(export / "vla.json", destination / "vla.json")
        copy_asset(export / "metadata.jsonl", destination / "vlm_metadata.jsonl")
        with np.load(export / "reference.npz", allow_pickle=False) as arrays:
            states, actions = arrays["states"], arrays["reference_actions"]
            measured = arrays["measured_states"]
            check = verify_rows(segments, rows, states, actions, len(actions), 5e-8)
            trajectory = {key: arrays[key].tolist() for key in arrays.files}
        dump(destination / "trajectory.json", trajectory)
        old_prefix = original["id"] + "/"
        prefix = destination.relative_to(output).as_posix() + "/"
        observations, storyboard, metas = [], [], []
        for segment, cached, meta in zip(segments, original["segments"], metadata):
            start, end = segment["start_frame"], segment["end_frame"]
            segment.update(state_at_start=states[start].tolist(), command_endpoint=actions[end].tolist(),
                           actions=actions[start:end + 1].tolist(),
                           padding_rows_if_chunked=segment["num_chunks"] * 8 - (end - start + 1))
            observations.append([image["path"].replace(old_prefix, prefix, 1)
                                 for image in cached["student_images"]])
            storyboard.append([{**shot, "path": shot["path"].replace(old_prefix, prefix, 1)}
                               for shot in cached["shots"]])
            metas.append({**meta, "image_roles": [f"{image['role']} {image['camera']}"
                                                   for image in cached["student_images"]]})
        native = Path(vla["native_arrays"])
        copy_asset(native, destination / "native.npz")
        times = (np.arange(len(measured)) / data["fps"]).tolist()
        split = next(entry["split"] for entry in data["selection"] if entry["episode_id"] == original["episode_id"])
        episode = {"id": f"agibot_{index:02d}", "dataset": "AgiBot", "task": original["task"],
                   "source_episode_id": original["episode_id"], "source": {"split": split, "file": str(export)},
                   "version": "run_v1 · 2026-10-15", "frame_count": len(actions),
                   "observation_count": len(measured), "state_layout": data["contract"]["state_layout"],
                   "camera_names": [camera["name"] for camera in original["cameras"]],
                   "segments": segments, "vlm": rows, "vlm_jsonl_text": raw, "vlm_metadata": metas,
                   "observations": observations, "storyboard": storyboard, "poster": storyboard[0][0]["path"],
                   "directory": destination.relative_to(output).as_posix(),
                   "videos": [{"name": camera["name"], "path": camera["video"].replace(old_prefix, prefix, 1)}
                              for camera in original["cameras"]],
                   "timestamps": times, "display_timestamps": times,
                   "time_basis": "30 fps · N 个实测观测对应 N−1 条参考动作，保留末帧观测",
                   "checks": {**check, "committed_exports_verified": True, "committed_rows_byte_identical": True,
                              "native_sha256": sha(native)},
                   "downloads": ["vlm.jsonl", "vla.json", "vlm_metadata.jsonl", "trajectory.json",
                                 "segmentation.json", "native.npz", "export/reference.npz"]}
        dump(destination / "VALIDATION.json", episode["checks"])
        result.append(episode)
        print("AgiBot ready", original["episode_id"], len(segments), flush=True)
    return result


def build_robotwin(source, output):
    if read(source / "TEXT_V5_COMPLETE.json").get("text_dataset_ready") is not True:
        raise ValueError("RoboTwin text_v5 completion marker is unavailable")
    metadata_file = source / "metadata/val_text_v5.jsonl"
    rows_file = source / "val_text_v5.jsonl"
    metadata = {index: [] for index in ROBOTWIN_IDS}
    wanted = {}
    for meta in jsonl(metadata_file):
        if meta["episode_index"] in metadata:
            wanted[meta["output_line"]] = meta
    records = {index: {} for index in ROBOTWIN_IDS}
    all_lines = {index: [] for index in ROBOTWIN_IDS}
    last_line = max(wanted)
    with rows_file.open() as stream:
        for number, raw in enumerate(stream, 1):
            if number in wanted:
                meta = wanted[number]
                if hashlib.sha256(raw.encode()).hexdigest() != meta["row_sha256"]:
                    raise ValueError("RoboTwin training row hash differs from text_v5 metadata")
                row = json.loads(raw)
                index = meta["episode_index"]
                all_lines[index].append(raw)
                metadata[index].append(meta)
                key = meta["segment_index"]
                if key not in records[index] or meta["frame_index"] < records[index][key][0]["frame_index"]:
                    records[index][key] = (meta, row, raw)
            if number >= last_line:
                break
    manifest_path = source.parent / "robotwin_no_cot/source_snapshots/action_segmentation.jsonl"
    manifests = {}
    for manifest in jsonl(manifest_path):
        index = manifest["global_episode_index"]
        if index in records:
            manifests[index] = manifest
        if len(manifests) == len(records):
            break
    result = []
    for number, index in enumerate(ROBOTWIN_IDS, 1):
        manifest = manifests[index]
        chosen = [records[index][i] for i in range(manifest["segment_count"])]
        first = chosen[0][0]
        annotation_path = Path(first["text_v5"]["vla_annotation_file"])
        if sha(annotation_path) != first["text_v5"]["vla_annotation_sha256"]:
            raise ValueError("RoboTwin VLA v5 annotations changed")
        local_key = first["text_v5"]["vla_local_episode_id"]
        vla = read(annotation_path)[local_key]
        if vla["source_global_episode_index"] != index or vla["task"] != manifest["task_instruction"]:
            raise ValueError("RoboTwin VLA source identity differs")
        raw_data = pq.read_table(manifest["source_paths"]["endpose_parquet"]).to_pydict()
        joint = pq.read_table(manifest["source_paths"]["shuffled_joint_parquet"]).to_pydict()
        states, actions = np.asarray(raw_data["observation.state"]), np.asarray(raw_data["action"])
        times = np.asarray(raw_data["timestamp"], dtype=float)
        times -= times[0]
        segments = copy.deepcopy(manifest["segments"])
        rows, raw_lines, metas, observations = [], [], [], []
        directory = output / "episodes" / f"robotwin_{number:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        for i, (segment, (meta, row, raw), annotation) in enumerate(zip(segments, chosen, vla["segments"])):
            start, end = segment["start_frame"], segment["end_frame"]
            if meta["frame_index"] != start:
                raise ValueError("RoboTwin complete episode lacks a true segment boundary sample")
            target = json.loads(row["messages"][-1]["content"])
            if any(annotation[key] != target[key] for key in FIELDS if key != "guide_action"):
                raise ValueError("RoboTwin current target differs from VLA aligned_v5")
            if (start, end) != (annotation["start_frame"], annotation["end_frame"]):
                raise ValueError("RoboTwin VLA boundaries differ")
            segment.update(target, valid_action_rows=end - start + 1,
                           padding_rows_if_chunked=target["num_chunks"] * 8 - (end - start + 1),
                           state_at_start=states[start].tolist(), command_endpoint=actions[end].tolist(),
                           actions=actions[start:end + 1].tolist(), native_joint_actions=joint["action"][start:end + 1])
            user = row["messages"][1]["content"]
            if i:
                previous = next(json.loads(line.removeprefix("Previous command: "))
                                for line in user.splitlines() if line.startswith("Previous command: "))
                if previous != json.loads(rows[i - 1]["messages"][-1]["content"]):
                    raise ValueError("RoboTwin previous command differs from the prior boundary target")
            shots, roles = [], []
            for image_index, image in enumerate(row["images"]):
                source_image = Path(image)
                destination = directory / "observations" / source_image.name
                copy_asset(source_image, destination)
                shots.append(destination.relative_to(output).as_posix())
                role = "history" if len(row["images"]) == 6 and image_index < 3 else "current"
                roles.append(f"{role} {('head', 'left wrist', 'right wrist')[image_index % 3]}")
            observations.append(shots)
            rows.append(row)
            raw_lines.append(raw)
            metas.append({**meta, "image_roles": roles})
        check = verify_rows(segments, rows, states, actions, manifest["all_frames"], 5.1e-5)
        (directory / "vlm.jsonl").write_text("".join(raw_lines))
        (directory / "all_training_rows.jsonl").write_text("".join(all_lines[index]))
        (directory / "vlm_metadata.jsonl").write_text("".join(json.dumps(meta, ensure_ascii=False) + "\n" for meta in metas))
        dump(directory / "all_training_metadata.json", metadata[index])
        dump(directory / "vla.json", {local_key: vla})
        dump(directory / "segmentation.json", manifest)
        dump(directory / "trajectory.json", {"endpose": raw_data, "joint": joint})
        dump(directory / "source.json", {"dataset": str(source), "text_file": str(rows_file),
                                        "vla_file": str(annotation_path), "vla_key": local_key,
                                        "source_manifest": str(manifest_path), "boundary_rows": chosen and [m[0] for m in chosen]})
        print("RoboTwin decode", index, manifest["task_name"], manifest["all_frames"], flush=True)
        streams, media, rates = camera_video_frames(manifest)
        video = directory / "preview.mp4"
        frames = encode_video(video, composite_frames(streams, "ROBOTWIN ALIGNED V5", segments))
        stories = []
        for segment in segments:
            story = []
            for frame_index in (segment["start_frame"], (segment["start_frame"] + segment["end_frame"]) // 2, segment["end_frame"]):
                path = directory / "storyboard" / f"{frame_index:06d}.jpg"
                path.parent.mkdir(exist_ok=True)
                image = Image.new("RGB", (960, 480))
                for camera_index, stream in enumerate(streams):
                    image.paste(stream[frame_index].resize((320, 240)), (camera_index * 320, 120))
                image.crop((0, 120, 960, 360)).save(path, quality=90)
                story.append({"path": path.relative_to(output).as_posix(), "frame": frame_index})
            stories.append(story)
        episode = {"id": f"robotwin_{number:02d}", "dataset": "RoboTwin", "task": vla["task"],
                   "source_episode_id": manifest["episode_id"], "source": {"split": "val", "file": str(rows_file)},
                   "version": "aligned_v5 · 2026-10-17", "frame_count": frames,
                   "state_layout": [f"{arm}_{key}" for arm in ("left", "right")
                                    for key in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")],
                   "camera_names": ["head", "left wrist", "right wrist"], "segments": segments,
                   "vlm": rows, "vlm_jsonl_text": "".join(raw_lines), "vlm_metadata": metas,
                   "observations": observations, "storyboard": stories, "poster": stories[0][0]["path"],
                   "directory": directory.relative_to(output).as_posix(),
                   "videos": [{"name": "三视角", "path": video.relative_to(output).as_posix()}],
                   "timestamps": times.tolist(), "display_timestamps": (np.arange(frames) / DISPLAY_FPS).tolist(),
                   "time_basis": f"回看 {DISPLAY_FPS} fps · 完整保留全部源帧，源轨迹时间戳独立展示",
                   "all_training_rows": len(all_lines[index]),
                   "checks": {**check, "source_rows_sha256_verified": True, "vla_v5_exact_subtask": True,
                              "history_v5_verified": True, "boundary_rows_byte_identical": True,
                              "all_episode_training_rows_preserved": len(all_lines[index]), "video_frames": frames,
                              "media_sources": media, "source_video_fps": rates},
                   "downloads": ["vlm.jsonl", "vla.json", "vlm_metadata.jsonl", "trajectory.json",
                                 "segmentation.json", "all_training_rows.jsonl", "source.json"]}
        dump(directory / "VALIDATION.json", episode["checks"])
        result.append(episode)
        print("RoboTwin ready", index, len(rows), "segments", len(all_lines[index]), "training rows", flush=True)
    return result


def overview(episodes, output):
    sheet = Image.new("RGB", (1440, len(episodes) * 220 + 50), "#f3f6fa")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
    draw.text((20, 15), "ABC-130k / AgiBot / RoboTwin | 4 complete episodes each", font=font, fill="#1c2e48")
    for index, episode in enumerate(episodes):
        top = 50 + index * 220
        draw.text((20, top), f"{episode['id']} | {episode['task'].split('|')[0][:115]}", font=font, fill="#1c2e48")
        shots = [episode["storyboard"][i][j] for i, j in ((0, 0), (len(episode["storyboard"]) // 2, 1), (-1, -1))]
        for column, shot in enumerate(shots):
            with Image.open(output / shot["path"]) as image:
                image.thumbnail((460, 180))
                sheet.paste(image, (20 + column * 480, top + 30))
    sheet.save(output / "overview.jpg", quality=92)


def render_page(data, output):
    template = (ROOT / "scripts/multi_dataset_episode_review.html").read_text()
    base = (ROOT / "scripts/episode_dataset_review.html").read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    base = base.replace("fonts/review-cjk.woff2", "fonts/review-cjk.woff").replace("format('woff2')", "format('woff')")
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__BASE_STYLE__", base).replace("__EPISODE_DATA__", embedded))
    for source, destination in (("pretrain-review-cjk.woff", "review-cjk.woff"),
                                ("latest-robotwin-cjk.woff", "review-latest.woff"),
                                ("agibot-completed-review-cjk.woff", "review-agibot.woff"),
                                ("OFL.txt", "OFL.txt")):
        copy_asset(ROOT / "scripts/assets" / source, output / "fonts" / destination)
    from fontTools import subset
    from fontTools.ttLib import TTFont

    font_path = Path("/tmp/ease-slides-tex/compat/usr/share/texlive/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if font_path.is_file():
        font = TTFont(font_path)
        options = subset.Options()
        options.flavor = "woff"
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes={ord(character) for character in template + embedded if ord(character) > 127})
        subsetter.subset(font)
        font.flavor = "woff"
        font.save(output / "fonts/review-extra.woff")
        font.close()
        documentation = font_path.parents[6] / "doc/texlive-doc/fonts/fandol"
        for name in ("README", "COPYING.gz"):
            if (documentation / name).is_file():
                copy_asset(documentation / name, output / "fonts/fandol" / name)
        copy_asset(font_path, output / "fonts/fandol/FandolHei-Regular.otf")
        (output / "fonts/fandol/MODIFICATION.txt").write_text(
            "review-extra.woff is a WOFF subset of FandolHei-Regular.otf, generated with fontTools "
            "for the Unicode characters present in this review. The unmodified source font, "
            "upstream README and COPYING.gz are supplied in this directory.\n"
        )


def package_review(output):
    data_validation = read(output / "VALIDATION.json")
    browser_validation = read(output / "BROWSER_VALIDATION.json")
    media_validation = read(output / "MEDIA_VALIDATION.json")
    if not all(report["passed"] for report in (data_validation, browser_validation, media_validation)):
        raise ValueError("All data, browser and media checks must pass before packaging")
    dump(output / "COMPLETE.json", {"status": "complete", "episodes": data_validation["episodes"],
                                   "segments": data_validation["planner_rows"],
                                   "action_rows": data_validation["action_rows"],
                                   "videos": len(media_validation["videos"]),
                                   "data_validation": True, "browser_validation": True,
                                   "media_validation": True, "offline": True})
    paths = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {path.relative_to(output).as_posix(): sha(path) for path in paths if path.name != "FILE_SHA256.json"}
    dump(output / "FILE_SHA256.json", manifest)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
        for name, expected in manifest.items():
            if hashlib.sha256(bundle.read(output.name + "/" + name)).hexdigest() != expected:
                raise ValueError(f"Archive file hash differs: {name}")
    archive.with_suffix(".zip.sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    print("ARCHIVE VERIFIED", archive, archive.stat().st_size, "bytes", len(manifest), "files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-review", type=Path, default=ROOT / "runs/abc130k_noncot_softchunks_128gpu_20261013_190459/intermediate_preview_20261014_001730")
    parser.add_argument("--agibot-review", type=Path, default=ROOT / "artifacts/agibot_generated_4_review")
    parser.add_argument("--robotwin", type=Path, default=Path("/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/datasets/robotwin_staged_stage1_noncot_20261015"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-page", action="store_true", help="Refresh only HTML and fonts from existing exported data")
    parser.add_argument("--package", action="store_true", help="Package an existing validated review")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.package:
        package_review(output)
        return
    if args.refresh_page:
        render_page(read(output / "complete_episodes.json"), output)
        print("PAGE REFRESHED", output / "index.html", flush=True)
        return
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.mkdir(parents=True)
    episodes = build_abc(args.abc_review.resolve(), output)
    episodes.extend(build_agibot(args.agibot_review.resolve(), output))
    episodes.extend(build_robotwin(args.robotwin.resolve(), output))
    counts = Counter(episode["dataset"] for episode in episodes)
    if counts != {"ABC-130k": 4, "AgiBot": 4, "RoboTwin": 4}:
        raise ValueError(f"Unexpected dataset selection: {counts}")
    report = {"passed": True, "episodes": 12, "dataset_counts": dict(counts),
              "planner_rows": sum(len(e["segments"]) for e in episodes),
              "action_rows": sum(e["frame_count"] for e in episodes),
              "results": [{"id": e["id"], "source_episode_id": e["source_episode_id"], **e["checks"]} for e in episodes]}
    dump(output / "VALIDATION.json", report)
    data = {"generated_at": datetime.now().isoformat(), "episodes": episodes, "validation": report}
    dump(output / "complete_episodes.json", data)
    dump(output / "SELECTION.json", [{"id": e["id"], "dataset": e["dataset"], "task": e["task"],
                                     "source_episode_id": e["source_episode_id"], "source": e["source"],
                                     "version": e["version"]} for e in episodes])
    render_page(data, output)
    overview(episodes, output)
    (output / "README.md").write_text(
        "# ABC-130k / AgiBot / RoboTwin：各四条完整 episode\n\n"
        "直接打开 index.html；所有视频、观测和字体都在本包中，保留相对目录即可离线查看。\n\n"
        "ABC 使用 2026-10-13 softchunks 生产中已提交分片；训练行与 COMMIT 哈希已重新核对。"
        "AgiBot 使用 run_v1 的四种已完成任务；导出文件逐一核对 COMMIT 哈希。"
        "RoboTwin 使用指定 20261015 Stage1 目录中 2026-10-17 发布的 val_text_v5.jsonl，"
        "并核对 aligned_v5 VLA 描述与历史 Previous command。\n\n"
        "每条展示全 episode 的分段起点训练样本，保留原始 system/user/assistant 和图片列表。"
        "RoboTwin 同时附 all_training_rows.jsonl，包含这四条 episode 全部段内起点增强训练行。"
        "原始训练行中的共享图片路径保留，页面用随包提供的相对图片路径。\n\n"
        "ABC guide 使用原始14维关节命令终点减状态起点；AgiBot 使用16维下一实测状态减当前状态；"
        "RoboTwin 使用16维 endpose action[end] - observation.state[start]。"
        "AgiBot保留末帧观测；RoboTwin回看视频沿用15fps，源轨迹时间戳另存，不混用。\n\n"
        "VALIDATION.json 保存动作全覆盖、guide、chunks、源文件和训练行校验结果；"
        "BROWSER_VALIDATION.json 保存浏览器检查。SELECTION.json 保存完整来源身份。\n"
    )
    print("COMPLETE", output / "index.html", report["planner_rows"], "segments", flush=True)


if __name__ == "__main__":
    main()
