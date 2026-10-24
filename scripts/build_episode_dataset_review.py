#!/usr/bin/env python3
"""Export complete, aligned VLM/VLA episodes and a portable offline review."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import av
import h5py
import ijson
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atomic_seg.recovery_cot.native import load_native_endpose

CAMERAS = ("head", "left_wrist", "right_wrist")
VIDEO_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
HDF5_CAMERAS = (
    "observation/head_camera/rgb",
    "observation/images/cam_left_wrist/rgb_jpeg",
    "observation/images/cam_right_wrist/rgb_jpeg",
)
COLORS = ("#3b82f6", "#8b5cf6", "#e09b24", "#10a78f", "#e4667a")
DISPLAY_FPS = 15
FONT = "/usr/share/fonts/dejavu/DejaVuSans.ttf"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line), line


def select_manifest(path, selected):
    records = {}
    for _, row, _ in read_jsonl(path):
        index = row["global_episode_index"]
        if index in selected:
            records[index] = row
        if len(records) == len(selected):
            break
    if set(records) != set(selected):
        raise ValueError(f"Missing episodes in {path}: {set(selected) - set(records)}")
    return records


def select_vla(path, selected):
    records = {}
    with path.open("rb") as stream:
        for key, value in ijson.kvitems(stream, "", use_float=True):
            index = value["global_episode_index"]
            if index in selected:
                records[index] = (key, value)
            if len(records) == len(selected):
                break
    if set(records) != set(selected):
        raise ValueError(f"Missing VLA episodes in {path}")
    return records


def select_vlm(root, selected, normal):
    records = {index: [] for index in selected}
    metadata = {index: [] for index in selected}
    provenance = {index: [] for index in selected}
    raw_lines = {index: [] for index in selected}
    if normal:
        wanted = {}
        for _, row, _ in read_jsonl(root / "metadata/row_mapping.jsonl"):
            if row["episode_index"] in selected:
                wanted[row["output_line"]] = row
        last = max(wanted)
        with (root / "data/vlm/train.jsonl").open() as vlm_stream, (
            root / "metadata/vlm_records.jsonl"
        ).open() as metadata_stream:
            for line_number, (line, meta_line) in enumerate(zip(vlm_stream, metadata_stream), 1):
                if line_number in wanted:
                    mapping = wanted[line_number]
                    index = mapping["episode_index"]
                    row, meta = json.loads(line), json.loads(meta_line)
                    assert meta["episode_index"] == index
                    assert meta["segment_index"] == mapping["segment_index"]
                    records[index].append(row)
                    metadata[index].append(meta)
                    provenance[index].append(mapping)
                    raw_lines[index].append(line)
                if line_number >= last:
                    break
    else:
        for line_number, row, line in read_jsonl(root / "data/vlm/train.jsonl"):
            index = row["episode_index"]
            if index in selected:
                records[index].append(row)
                metadata[index].append(row)
                provenance[index].append({"source_line": line_number, "segment_index": row["segment_index"]})
                raw_lines[index].append(line)
            if index > max(selected):
                break
    return records, metadata, provenance, raw_lines


def validate_episode(manifest, vla, rows, metadata):
    count = len(manifest["segments"])
    assert count == manifest["segment_count"] == len(vla["segments"]) == len(rows) == len(metadata)
    assert count in (4, 5)
    assert vla["all_frames"] == manifest["all_frames"]
    assert vla["global_episode_index"] == manifest["global_episode_index"]
    assert vla["source_episode_id"] == manifest["episode_id"]
    assert "messages" not in vla and "system" not in vla
    cursor = 0
    for i, (segment, action, row, meta) in enumerate(zip(manifest["segments"], vla["segments"], rows, metadata)):
        assert segment["segment_index"] == action["source_segment_id"] == meta["segment_index"] == i
        assert segment["start_frame"] == action["start_frame"] == meta["segment_start_frame"] == cursor
        assert segment["end_frame"] == action["end_frame"] == meta["segment_end_frame"]
        assert segment["global_block_index"] == action["global_block_index"]
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"] == meta["messages"] and row["images"] == meta["images"]
        assert row["messages"][0]["content"]
        assert row["messages"][1]["content"].count("<image>") == len(row["images"])
        assert all(Path(image).is_file() for image in row["images"])
        target = json.loads(row["messages"][2]["content"].split("</think>")[-1].strip())
        assert target == meta["target"]
        for key in ("guide_action", "primary_action_verb", "sub_task", "num_chunks"):
            assert target[key] == action[key], (i, key)
            if key in segment:
                assert target[key] == segment[key]
        assert target["num_chunks"] == (segment["end_frame"] - cursor + 8) // 8
        if i:
            assert meta["previous_boundary_frame"] == manifest["segments"][i - 1]["start_frame"]
        cursor = segment["end_frame"] + 1
    assert cursor == manifest["all_frames"], "Episode has missing final frames"


def camera_video_frames(manifest):
    parquet = Path(manifest["source_paths"]["official_parquet"])
    root = parquet.parents[2]
    streams, sources, rates = [], [], []
    for camera in VIDEO_CAMERAS:
        path = root / "videos" / parquet.parent.name / f"observation.images.{camera}" / f"{parquet.stem}.mp4"
        with av.open(str(path)) as container:
            rates.append(float(container.streams.video[0].average_rate))
            frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
        assert len(frames) == manifest["all_frames"], path
        streams.append(frames)
        sources.append(str(path))
    assert len(set(rates)) == 1
    return streams, sources, rates[0]


def camera_hdf5_frames(path):
    with h5py.File(path, "r") as source:
        return [[Image.open(io.BytesIO(bytes(np.asarray(value, dtype=np.uint8)))).convert("RGB")
                 for value in source[key]] for key in HDF5_CAMERAS]


def composite_frames(streams, label, segments=None):
    font = ImageFont.truetype(FONT, 15)
    count = len(streams[0])
    assert all(len(stream) == count for stream in streams)
    for index, views in enumerate(zip(*streams)):
        canvas = Image.new("RGB", (960, 544), "#101a2c")
        for view, size, xy in zip(views, ((640, 480), (320, 240), (320, 240)), ((0, 32), (640, 32), (640, 272))):
            canvas.paste(view.resize(size, Image.Resampling.LANCZOS), xy)
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{label} | frame {index}/{count-1} | HEAD + LEFT WRIST / RIGHT WRIST", font=font, fill="white")
        if segments:
            active = next(i for i, s in enumerate(segments) if s["start_frame"] <= index <= s["end_frame"])
            draw.rectangle((0, 512, 960, 544), fill=COLORS[active % len(COLORS)])
            draw.text((12, 520), f"A{active+1:02}  {segments[active]['primary_action_verb']}", font=font, fill="white")
        else:
            draw.text((12, 520), "Paired source trajectory: failure context", font=font, fill="#fbbf24")
        yield canvas


def encode_video(path, frames):
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=DISPLAY_FPS)
        stream.width, stream.height, stream.pix_fmt = 960, 544, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast", "threads": "2"}
        count = 0
        for count, frame in enumerate(frames, 1):
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == count, (path, decoded, count)
    return count


def copy_observations(rows, directory, output):
    images, mapping = [], {}
    for row in rows:
        local_images = []
        for source in row["images"]:
            if source not in mapping:
                path = Path(source)
                name = hashlib.sha256(source.encode()).hexdigest()[:12] + "_" + path.name
                target = directory / name
                shutil.copy2(path, target)
                with Image.open(target) as image:
                    image.verify()
                assert sha(target) == sha(path)
                mapping[source] = str(target.relative_to(output))
            local_images.append(mapping[source])
        images.append(local_images)
    return images, mapping


def build_episode(kind, manifest, vla_entry, rows, metadata, provenance, raw_lines, root, output):
    key, vla = vla_entry
    validate_episode(manifest, vla, rows, metadata)
    index = manifest["global_episode_index"]
    slug = f"{kind}_{index:06d}_{manifest['task_name']}"
    directory = output / "episodes" / slug
    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    observed, mapping = copy_observations(rows, images_dir, output)
    dump(directory / "vla.json", {key: vla})
    (directory / "vlm.jsonl").write_text("".join(raw_lines))
    dump(directory / "vlm.json", rows)
    dump(directory / "vlm_metadata.json", metadata)
    dump(directory / "segmentation.json", manifest)
    dump(directory / "source_rows.json", provenance)
    dump(directory / "asset_mapping.json", mapping)
    assert json.loads((directory / "vla.json").read_text())[key] == vla
    assert [r for _, r, _ in read_jsonl(directory / "vlm.jsonl")] == rows

    context = None
    if kind == "normal":
        streams, media_sources, source_video_fps = camera_video_frames(manifest)
        endpose = pq.read_table(manifest["source_paths"]["endpose_parquet"]).to_pydict()
        joints = pq.read_table(manifest["source_paths"]["shuffled_joint_parquet"]).to_pydict()
        trajectory = {"endpose_parquet": endpose, "joint_parquet": joints}
        states, actions = np.asarray(endpose["observation.state"]), np.asarray(endpose["action"])
        timestamps = np.asarray(endpose["timestamp"])
    else:
        paths = manifest["source_paths"]
        hdf5_path = Path(paths["recovery_hdf5"])
        states, actions, timestamps, extras = load_native_endpose(hdf5_path)
        streams = camera_hdf5_frames(hdf5_path)
        context = {name: json.loads(Path(paths[name]).read_text())
                   for name in ("source_episode_json", "recovery_episode_json", "pair_audit")}
        assert context["recovery_episode_json"]["recovery_success"] is True
        assert context["pair_audit"]["ok"] is True
        with h5py.File(hdf5_path, "r") as source:
            trajectory = {"state_layout": metadata[0]["state_layout"],
                          "endpose_state": states.tolist(), "endpose_next_state": actions.tolist(),
                          "native_action": np.asarray(source["action"]).tolist(),
                          "joint_action_vector": np.asarray(source["joint_action/vector"]).tolist(),
                          "timestamp": timestamps.tolist(), "transition_extra": extras}
        source_video_fps = context["recovery_episode_json"]["fps"]
        media_sources = [str(hdf5_path)]
        source_hdf5 = Path(paths["source_episode_json"]).with_name("episode.hdf5")
        source_streams = camera_hdf5_frames(source_hdf5)
        source_count = encode_video(directory / "failure_context.mp4", composite_frames(source_streams, "SOURCE FAILURE"))
        def pair_frames():
            yield from composite_frames(source_streams, "SOURCE FAILURE")
            yield from composite_frames(streams, "RECOVERY", manifest["segments"])
        pair_count = encode_video(directory / "failure_to_recovery.mp4", pair_frames())
        assert pair_count == source_count + manifest["all_frames"]
        context["preview_media"] = {"source_frames": source_count, "pair_frames": pair_count,
                                    "source_hdf5": str(source_hdf5)}
        dump(directory / "recovery_context.json", context)

    assert len(states) == len(actions) == len(timestamps) == manifest["all_frames"]
    max_delta_error = 0.0
    for segment, meta in zip(manifest["segments"], metadata):
        start, end = segment["start_frame"], segment["end_frame"]
        assert np.allclose(states[start], meta["current_state"], atol=1e-6, rtol=0)
        assert np.allclose(actions[end], meta["guide_action_target_state"], atol=1e-6, rtol=0)
        error = float(np.max(np.abs(actions[end] - states[start] - np.asarray(meta["target"]["guide_action"]))))
        assert error <= 0.000051, (slug, start, error)
        max_delta_error = max(max_delta_error, error)
    dump(directory / "trajectory.json", trajectory)
    encoded = encode_video(directory / "episode.mp4", composite_frames(streams, kind.upper(), manifest["segments"]))
    assert encoded == manifest["all_frames"]
    next(composite_frames(streams, kind.upper(), manifest["segments"])).save(directory / "poster.jpg", quality=90)
    story = []
    for i, segment in enumerate(manifest["segments"]):
        start, end = segment["start_frame"], segment["end_frame"]
        relative = []
        for label, frame in (("start", start), ("middle", (start + end) // 2), ("end", end)):
            path = images_dir / f"story_{i}_{label}.jpg"
            streams[0][frame].save(path, quality=88)
            relative.append(str(path.relative_to(output)))
        story.append(relative)
    relative = str(directory.relative_to(output))
    episode = {"id": slug, "kind": kind, "episode_index": index, "episode_id": manifest["episode_id"],
               "task_name": manifest["task_name"], "task": manifest["task_instruction"],
               "segment_count": len(manifest["segments"]), "frame_count": manifest["all_frames"],
               "display_fps": DISPLAY_FPS, "source_video_fps": source_video_fps,
               "timestamps": timestamps.tolist(), "video": relative + "/episode.mp4",
               "poster": relative + "/poster.jpg", "directory": relative, "manifest": manifest,
               "vla_key": key, "vla": vla, "vlm": rows, "vlm_jsonl_text": "".join(raw_lines), "vlm_metadata": metadata,
               "source_rows": provenance, "observations": observed, "storyboard": story,
               "recovery_context": context, "source_root": str(root), "media_sources": media_sources,
               "trajectory": trajectory,
               "checks": {"complete_episode": True, "all_roles_preserved": True,
                          "vlm_vla_segments_aligned": True, "raw_state_target_checked": True,
                          "max_guide_rounding_error": max_delta_error,
                          "video_frames": encoded, "copied_observation_images": len(mapping)}}
    print(f"built {slug}: {len(rows)} segments / {encoded} frames", flush=True)
    return episode


def overview(episodes, output):
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 13)
    canvas = Image.new("RGB", (1540, 90 + len(episodes) * 210), "#eff3f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "COMPLETE EPISODES  |  4 NORMAL + 4 RECOVERY  |  37 ATOMIC ACTIONS", font=font, fill="#152846")
    draw.text((28, 52), "Each row is a complete episode. Images show atomic-action starting frames; full messages and videos are in index.html.", font=small, fill="#58708a")
    for row, episode in enumerate(episodes):
        y = 90 + row * 210
        accent = "#257654" if episode["kind"] == "normal" else "#b25b12"
        draw.rounded_rectangle((16, y, 1524, y + 196), radius=10, fill="white")
        draw.text((30, y + 12), f"{episode['kind'].upper()} {episode['episode_index']}", font=font, fill=accent)
        draw.text((30, y + 40), episode["task_name"], font=small, fill="#263c58")
        draw.text((30, y + 64), f"{episode['segment_count']} actions / {episode['frame_count']} frames", font=small, fill="#58708a")
        if episode["kind"] == "recovery":
            draw.text((30, y + 88), f"L{episode['manifest']['recovery_level']}", font=font, fill=accent)
        for i, segment in enumerate(episode["manifest"]["segments"]):
            x = 300 + i * 240
            with Image.open(output / episode["storyboard"][i][0]) as image:
                canvas.paste(image.resize((224, 148)), (x, y + 12))
            draw.rectangle((x, y + 160, x + 224, y + 164), fill=COLORS[i])
            draw.text((x, y + 168), f"A{i+1:02} [{segment['start_frame']}-{segment['end_frame']}]", font=small, fill="#263c58")
    canvas.save(output / "overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for kind in ("normal", "recovery"):
        root = Path(config[f"{kind}_root"])
        selected = config[f"{kind}_episode_indices"]
        assert len(selected) == len(set(selected)) == 4
        manifest_path = root / "metadata/source_manifest.jsonl" if kind == "normal" else Path(config["recovery_manifest"])
        manifests = select_manifest(manifest_path, selected)
        vla = select_vla(root / "data/vla/train.json", selected)
        rows, metadata, provenance, raw_lines = select_vlm(root, selected, kind == "normal")
        for index in selected:
            episodes.append(build_episode(kind, manifests[index], vla[index], rows[index], metadata[index],
                                          provenance[index], raw_lines[index], root, output))
    sources_dir = output / "sources"
    sources_dir.mkdir(exist_ok=True)
    for source, name in ((Path(config["normal_root"]) / "COMPLETE.json", "normal_COMPLETE.json"),
                         (Path(config["recovery_root"]) / "COT_STYLE_DIVERSIFICATION_REPORT.json", "recovery_COT_STYLE_REPORT.json"),
                         (Path(config["recovery_manifest"]).parents[1] / "SEGMENTATION_REFINEMENT_REPORT.json", "recovery_SEGMENTATION_REPORT.json")):
        shutil.copy2(source, sources_dir / name)
    payload = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "episodes": episodes,
               "sources": {"normal": "2026-10-03 最终修复版；2026-10-04 混合集引用同一 CoT 源",
                           "recovery": "2026-09-27 CoT 五风格版；2026-09-21 still-merged 划分"},
               "counts": {"normal": 4, "recovery": 4, "atomic_actions": sum(e["segment_count"] for e in episodes)}}
    dump(output / "complete_episodes.json", payload)
    template = Path(__file__).with_name("episode_dataset_review.html").read_text()
    if config.get("font_file"):
        font_dir = output / "fonts"
        font_dir.mkdir(exist_ok=True)
        shutil.copy2(config["font_file"], font_dir / "review-cjk.woff2")
        shutil.copy2(config["font_license"], font_dir / "OFL.txt")
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output / "index.html").write_text(template.replace("__EPISODE_DATA__", embedded))
    overview(episodes, output)
    validation = {"passed": True, "episode_count": len(episodes), "counts": payload["counts"],
                  "episodes": [{"id": e["id"], **e["checks"]} for e in episodes],
                  "source_data_modified": False, "vlm_original_rows_preserved": True,
                  "vla_original_episode_objects_preserved": True,
                  "recovery_levels": dict(Counter(e["manifest"].get("recovery_level") for e in episodes if e["kind"] == "recovery"))}
    dump(output / "VALIDATION.json", validation)
    (output / "README.md").write_text(
        "# 完整 episode 划分可视化\n\n解压后直接打开 `index.html`，无需联网或启动服务器。\n\n"
        "4 条正常 + 4 条 recovery；每条 4–5 个原子动作，共 37 段。\n\n"
        "正常源：2026-10-03 最终修复版。Recovery 源：2026-09-27 CoT 版，使用 2026-09-21 still-merged 划分。"
        "Recovery 原有 system、措辞、still 段全部保留，不因新代码合同而重生成。\n\n"
        "每条包含三视角完整视频、全部动作时间轴、完整 VLM system/user/assistant、全部输入图片、完整 VLA episode JSON、"
        "元数据、状态/动作轨迹及来源映射。VLA 源格式没有 system/messages。Recovery 另含完整失败→恢复视频及原始上下文。\n\n"
        "视频按 15 fps 展示，逐帧保留；恢复源数据为约 50 Hz，因此显示速度慢于源时间。正常源视频为 15 fps，"
        "状态数据约 50 Hz。时间轴以原始 action row/frame index 对齐，源时间戳独立显示。\n\n"
        "`episodes/*/vlm.jsonl` 保留源行文本；`vla.json` 保留完整源 episode 对象及原 key。"
        "原始 JSON 中图片绝对路径保留；页面使用随包复制的图片，映射在 asset_mapping.json。\n\n"
        "`overview.png` 是总览；`complete_episodes.json` 汇总完整数据；`VALIDATION.json` 记录对齐与媒体检查。\n\n"
        "复现：`/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python scripts/build_episode_dataset_review.py "
        "--config artifacts/episode_review_20261009/selection.json`\n"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.json")
    dump(output / "checksums.json", {str(path.relative_to(output)): sha(path) for path in files})
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as package:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                package.write(path, str(Path(output.name) / path.relative_to(output)))
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
    print(json.dumps({"html": str(output / "index.html"), "zip": str(archive),
                      "zip_bytes": archive.stat().st_size, "validation": validation}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
