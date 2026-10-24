#!/usr/bin/env python3
"""Render the historical October 12, 2026 WJH accepted-episode preview."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_seg.common.locking import writer_lock
from atomic_seg.recovery_cot.native import load_native_endpose
from scripts.build_episode_dataset_review import (
    DISPLAY_FPS, camera_hdf5_frames, camera_video_frames,
    composite_frames, copy_observations, dump, encode_video,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET_KEYS = ("guide_action", "primary_action_verb", "sub_task", "num_chunks")


def target(row):
    answer = row["messages"][-1]["content"]
    if answer.count("<think>") != 1 or answer.count("</think>") != 1:
        raise ValueError("expected one nonempty CoT block")
    before, final = answer.split("</think>", 1)
    if not before.split("<think>", 1)[1].strip():
        raise ValueError("empty CoT")
    result = json.loads(final)
    if set(result) != set(TARGET_KEYS):
        raise ValueError("unexpected final answer keys")
    return result


def named_state(value):
    return [x for arm in ("left", "right") for x in (
        *value[arm]["xyz"], *value[arm]["quat_wxyz"], value[arm]["gripper"]
    )]


def source_line(ref):
    with Path(ref["path"]).open("rb") as stream:
        stream.seek(ref["offset"])
        raw = stream.read(ref["bytes"])
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("accepted source row changed")
    return raw


def export_receipt(receipt, destination):
    if receipt.get("journal"):
        source = Path(receipt["journal"])
        with source.open("rb") as stream:
            stream.seek(receipt["byte_offset"])
            raw = stream.read(receipt["bytes"])
    elif receipt.get("path") and Path(receipt["path"]).is_file():
        source = Path(receipt["path"])
        raw = source.read_bytes()
    else:
        raise FileNotFoundError(f"generation receipt unavailable: {receipt}")
    value = json.loads(raw)
    dump(destination, {"source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "receipt": receipt, "request_and_response": value})
    return value


def audit_and_provenance(item, index, output, directory):
    source = output / "work/model_audits" / f"{item['kind']}_{item['index']:06d}_{index:03d}.json"
    audit = json.loads(source.read_text())
    result, row = audit["result"], item["rows"][index]
    if result["status"] != "ACCEPTED" or result["new_row"] != row:
        raise ValueError("preview does not match accepted result")
    accepted = next(a for a in result["attempts"]
                    if a.get("stage") == "write_and_audit" and a.get("attempt") == result["accepted_attempt"])
    critics = accepted["critics"]
    if set(critics) != {"grounding_critic", "action_critic"} or any(v["review"]["verdict"] != "PASS" for v in critics.values()):
        raise ValueError("both source critics must pass")
    job = result["job"]
    if item["kind"] == "normal":
        with Path(job["source_path"]).open("rb") as stream:
            stream.seek(job["source_offset"])
            raw = stream.readline()
        if hashlib.sha256(raw).hexdigest() != job["source_raw_sha256"]:
            raise ValueError("original normal input hash changed")
        original = json.loads(raw)
        if target(original) != target(row):
            raise ValueError("normal target was modified by rendering")
    else:
        original = json.loads((output / "work/source_jobs" / f"{result['id']}.json").read_text())["job"]["row"]
    if original["messages"][:2] != row["messages"][:2] or original["images"] != row["images"]:
        raise ValueError("student system/user/images do not match source")
    relative = directory.relative_to(output) / "model_audits" / f"segment_{index:03d}.json"
    dump(output / relative, audit)
    requests = []
    receipts = [("observer", result["observer_receipt"]), ("writer", accepted["writer_receipt"])]
    receipts += [(name, value["receipt"]) for name, value in critics.items()]
    for stage, receipt in receipts:
        path = directory / "model_audits" / f"segment_{index:03d}_{stage}.json"
        request = export_receipt(receipt, path)
        requests.append({"stage": stage, "path": str(path.relative_to(output)),
                         "model": request.get("parameters", {}).get("model", "Qwen/Qwen3.8-Flash-Next")})
    ref = item["accepted_refs"][index]
    round_name = "第二轮补跑" if "round2" in ref["path"] else "掉线补跑" if "postpass" in ref["path"] else "首轮生成"
    return {"status": "ACCEPTED", "generation_round": round_name, "style": job["style"],
            "accepted_attempt": result["accepted_attempt"], "finished_at": result.get("finished_at"),
            "critics": {name: value["review"]["verdict"] for name, value in critics.items()},
            "source_row": ref, "source_result": audit["source"],
            "result_path": str(relative), "requests": requests,
            "raw_assistant_sha256": hashlib.sha256(row["messages"][-1]["content"].encode()).hexdigest()}


def load_trajectory(item):
    manifest = item["manifest"]
    if item["kind"] == "normal":
        streams, media, fps = camera_video_frames(manifest)
        data = pq.read_table(manifest["source_paths"]["endpose_parquet"]).to_pydict()
        joint = pq.read_table(manifest["source_paths"]["shuffled_joint_parquet"]).to_pydict()
        states, actions, timestamps = np.asarray(data["observation.state"]), np.asarray(data["action"]), np.asarray(data["timestamp"])
        trajectory = {"endpose_parquet": data, "joint_parquet": joint}
        native = joint["action"]
    else:
        path = Path(manifest["source_paths"]["recovery_hdf5"])
        states, actions, timestamps, extras = load_native_endpose(path)
        streams = camera_hdf5_frames(path)
        with h5py.File(path, "r") as f:
            native = np.asarray(f["action"]).tolist()
            trajectory = {"endpose_state": states.tolist(), "endpose_next_state": actions.tolist(),
                          "native_action": native, "joint_action_vector": np.asarray(f["joint_action/vector"]).tolist(),
                          "timestamp": timestamps.tolist(), "transition_extra": extras}
        info = json.loads(Path(manifest["source_paths"]["recovery_episode_json"]).read_text())
        media, fps = [str(path)], info["fps"]
    return states, actions, timestamps, native, streams, media, fps, trajectory


def validate(item, states, actions, timestamps):
    manifest, rows, metadata = item["manifest"], item["rows"], item["source_metadata"]
    if len(rows) != manifest["segment_count"] or len(metadata) != len(rows):
        raise ValueError("missing episode CoT rows")
    n = manifest["all_frames"]
    if len(states) != len(actions) or len(states) != n or len(timestamps) != n:
        raise ValueError("raw trajectory length mismatch")
    cursor, error = 0, 0.0
    for i, (s, row, meta) in enumerate(zip(manifest["segments"], rows, metadata)):
        start, end = s["start_frame"], s["end_frame"]
        if start != cursor or end < start or s["segment_index"] != i:
            raise ValueError("gapped or overlapping source segmentation")
        answer = target(row)
        if answer != meta["target"] or any(answer[k] != s[k] for k in TARGET_KEYS if k in s):
            raise ValueError("manifest/CoT/metadata target mismatch")
        if answer["num_chunks"] != (end - start + 8) // 8:
            raise ValueError("num_chunks differs from inclusive native action rows")
        local_error = float(np.max(abs(actions[end] - states[start] - np.asarray(answer["guide_action"]))))
        if local_error > .000051:
            raise ValueError(f"raw guide mismatch at {start}: {local_error}")
        error = max(error, local_error)
        user = row["messages"][1]["content"]
        current = next(json.loads(line[len("Current state: "):]) for line in user.splitlines() if line.startswith("Current state: "))
        if not np.allclose(named_state(current), states[start], atol=.000051, rtol=0):
            raise ValueError("visible current state differs from raw source")
        expected_frames = [start] * 3 if i == 0 else [manifest["segments"][i - 1]["start_frame"]] * 3 + [start] * 3
        frames = [int(re.search(r"_f(\d+)", Path(p).name).group(1)) for p in row["images"]]
        if frames != expected_frames or user.count("<image>") != len(frames):
            raise ValueError("student current/history image alignment mismatch")
        if i:
            previous = next(json.loads(line[len("Previous command: "):]) for line in user.splitlines() if line.startswith("Previous command: "))
            if previous != target(rows[i - 1]):
                raise ValueError("previous command does not match prior segment")
        if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
            raise ValueError("unexpected message roles")
        if item["kind"] == "normal":
            if meta["segment_start_frame"] != start or meta["segment_end_frame"] != end:
                raise ValueError("normal metadata boundaries differ")
        elif meta["frame_index"] != start or meta["action_end_frame_inclusive"] != end:
            raise ValueError("recovery metadata boundaries differ")
        cursor = end + 1
    if cursor != n:
        raise ValueError("incomplete final episode coverage")
    return error


def build_episode(item, output):
    manifest, rows = item["manifest"], item["rows"]
    slug = f"{item['kind']}_{item['index']:06d}_{item['task']}"
    directory = output / "episodes" / slug
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    for ref, row, line in zip(item["accepted_refs"], rows, item["raw_lines"]):
        raw = source_line(ref)
        if raw.decode() != line or json.loads(raw) != row:
            raise ValueError("accepted source row changed since selection")
    states, actions, timestamps, native, streams, media, fps, trajectory = load_trajectory(item)
    guide_error = validate(item, states, actions, timestamps)
    observations, mapping = copy_observations(rows, images, output)
    (directory / "vlm.jsonl").write_text("".join(item["raw_lines"]))
    dump(directory / "vlm.json", rows)
    dump(directory / "vlm_metadata.json", item["source_metadata"])
    dump(directory / "segmentation.json", manifest)
    dump(directory / "trajectory.json", trajectory)
    dump(directory / "asset_mapping.json", mapping)
    if item["kind"] == "normal":
        key, vla = item["vla_entry"]
        vla_origin = "原始 VLA episode 对象，逐字段保留"
    else:
        key = str(item["index"])
        vla = {"schema": "review_reconstructed_action_episode", "global_episode_index": item["index"],
               "source_episode_id": manifest["episode_id"], "task": manifest["task_instruction"],
               "all_frames": manifest["all_frames"], "segments": copy.deepcopy(manifest["segments"]),
               "provenance": "rebuilt for review from current recovery manifest targets; native actions are in trajectory.json"}
        vla_origin = "本轮 Recovery 未导出独立 VLA：依据当前 manifest 的动作目标整理；原始动作另附"
    dump(directory / "vla.json", {key: vla})
    native_segments = []
    for s in manifest["segments"]:
        start, end = s["start_frame"], s["end_frame"]
        native_segments.append({"start_frame": start, "end_frame": end,
                                "native_actions": native[start:end + 1],
                                "endpose_states": states[start:end + 1].tolist(),
                                "endpose_action_targets": actions[start:end + 1].tolist(),
                                "timestamps": timestamps[start:end + 1].tolist()})
    if [a for s in native_segments for a in s["native_actions"]] != native:
        raise ValueError("native action row reconstruction mismatch")
    dump(directory / "native_actions_by_segment.json", native_segments)
    receipts = [audit_and_provenance(item, i, output, directory) for i in range(len(rows))]
    dump(directory / "source_rows.json", receipts)
    count = encode_video(directory / "episode.mp4", composite_frames(streams, item["kind"].upper(), manifest["segments"]))
    if count != manifest["all_frames"]:
        raise ValueError("video did not preserve every frame")
    next(composite_frames(streams, item["kind"].upper(), manifest["segments"])).save(directory / "poster.jpg", quality=92)
    storyboard = []
    for i, s in enumerate(manifest["segments"]):
        paths = []
        for label, frame in (("start", s["start_frame"]), ("middle", (s["start_frame"] + s["end_frame"]) // 2), ("end", s["end_frame"])):
            canvas = next(composite_frames([[camera[frame]] for camera in streams], f"SOURCE ROW {frame}"))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 512, 960, 544), fill="#20385d")
            draw.text((12, 520), f"SEGMENT A{i+1:02d} | {label.upper()} | SOURCE ROW {frame}",
                      font=ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15), fill="white")
            path = images / f"story_{i}_{label}.jpg"
            canvas.save(path, quality=89)
            paths.append(str(path.relative_to(output)))
        storyboard.append(paths)
    context = None
    if item["kind"] == "recovery":
        context = {name: json.loads(Path(manifest["source_paths"][name]).read_text())
                   for name in ("source_episode_json", "recovery_episode_json", "pair_audit")}
        source_hdf5 = Path(manifest["source_paths"]["source_episode_json"]).with_name("episode.hdf5")
        source_streams = camera_hdf5_frames(source_hdf5)
        def pair():
            yield from composite_frames(source_streams, "SOURCE FAILURE")
            yield from composite_frames(streams, "RECOVERY", manifest["segments"])
        pair_count = encode_video(directory / "failure_to_recovery.mp4", pair())
        context["preview_media"] = {"source_frames": len(source_streams[0]), "pair_frames": pair_count,
                                    "source_hdf5": str(source_hdf5)}
        dump(directory / "recovery_context.json", context)
    relative = str(directory.relative_to(output))
    lengths = [s["end_frame"] - s["start_frame"] + 1 for s in manifest["segments"]]
    episode = {"id": slug, "kind": item["kind"], "episode_index": item["index"], "episode_id": manifest["episode_id"],
               "task_name": item["task"], "task": manifest["task_instruction"], "frame_count": manifest["all_frames"],
               "segment_count": len(rows), "display_fps": DISPLAY_FPS, "source_video_fps": fps,
               "video": relative + "/episode.mp4", "poster": relative + "/poster.jpg", "directory": relative,
               "timestamps": timestamps.tolist(), "manifest": manifest, "vla_key": key, "vla": vla,
               "vla_origin": vla_origin, "vlm": rows, "vlm_jsonl_text": "".join(item["raw_lines"]),
               "vlm_metadata": item["source_metadata"], "observations": observations, "storyboard": storyboard,
               "source_rows": receipts, "native_action_segments": native_segments, "recovery_context": context,
               "source_root": sorted({str(Path(ref["path"]).parents[2]) for ref in item["accepted_refs"]}),
               "media_sources": media, "trajectory": trajectory,
               "length_stats": {"minimum": min(lengths), "maximum": max(lengths), "median": float(np.median(lengths)),
                                "chunks": dict(sorted(Counter((n + 7) // 8 for n in lengths).items()))},
               "checks": {"complete_episode": True, "raw_cot_preserved": True, "latest_accepted_rows_verified": True,
                          "source_system_user_images_preserved": True, "both_source_critics_pass": True,
                          "native_actions_preserved": True, "video_frames": count, "max_guide_rounding_error": guide_error}}
    dump(directory / "review_episode.json", episode)
    print("built", slug, len(rows), "CoT rows", count, "frames", flush=True)
    return episode


def overview(episodes, output):
    image = Image.new("RGB", (1400, 60 + 210 * len(episodes)), "#f3f6fa")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 13)
    draw.text((20, 18), f"ROBOTWIN CoT | {len(episodes)} COMPLETE EPISODES | {sum(e['segment_count'] for e in episodes)} ACTIONS", font=font, fill="#183556")
    for i, episode in enumerate(episodes):
        y = 60 + 210 * i
        draw.text((16, y + 8), f"{episode['kind']} #{episode['episode_index']}", font=font, fill="#183556")
        draw.text((16, y + 35), episode["task_name"], font=small, fill="#58708a")
        draw.text((16, y + 58), f"{episode['segment_count']} CoT rows / {episode['frame_count']} frames", font=small, fill="#58708a")
        for j, index in enumerate([0, len(episode["storyboard"]) // 2, len(episode["storyboard"]) - 1]):
            with Image.open(output / episode["storyboard"][index][0]) as im:
                im.thumbnail((350, 180))
                image.paste(im, (320 + 355 * j, y))
    image.save(output / "overview.jpg", quality=91)


def render(output, episodes):
    counts = dict(Counter(e["kind"] for e in episodes))
    counts["atomic_actions"] = sum(e["segment_count"] for e in episodes)
    sources = {"normal": "WJH visual v2：2026-10-10 首轮 + 第二轮补跑；第二轮完成 2026-10-12 00:49",
               "recovery": "WJH visual v2：2026-10-12 首轮 + 第二轮；第二轮完成 2026-10-12 17:33"}
    data = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sources": sources,
            "counts": counts, "episodes": episodes, "preview_scope": "complete episodes with every segment accepted; not a representative acceptance-rate sample"}
    dump(output / "complete_episodes.json", data)
    template = (ROOT / "scripts/latest_robotwin_cot_review.html").read_text()
    base = (ROOT / "scripts/episode_dataset_review.html").read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    base = base.replace("fonts/review-cjk.woff2", "fonts/latest-robotwin-cjk.woff").replace("format('woff2')", "format('woff')")
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__BASE_STYLE__", base).replace("__EPISODE_DATA__", embedded))
    (output / "fonts").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "scripts/assets/latest-robotwin-cjk.woff", output / "fonts/latest-robotwin-cjk.woff")
    shutil.copyfile(ROOT / "scripts/assets/OFL.txt", output / "fonts/OFL.txt")
    overview(episodes, output)
    validation = {"passed": True, "counts": counts, "episode_count": len(episodes),
                  "native_frames": sum(e["frame_count"] for e in episodes),
                  "episodes": [{"id": e["id"], **e["checks"]} for e in episodes],
                  "generation_rounds": dict(Counter(s["generation_round"] for e in episodes for s in e["source_rows"])),
                  "source_data_modified": False, "new_cot_generated_by_review": False,
                  "selection": "every segment in each selected episode is ACCEPTED; not a whole-dataset quality claim"}
    dump(output / "VALIDATION.json", validation)
    (output / "README.md").write_text(
        "# RoboTwin 历史 CoT · 2026-10-12 完整 episode 审阅\n\n"
        "解压后打开 index.html。4 条正常、4 条 Recovery，共 43 段。完整三相机视频、全部动作、原始学生图片、完整 system/user/CoT assistant、逐段 native action 与来源审核均在页面。\n\n"
        + "\n\n".join(sources.values())
        + "\n\n本页只读导出刚生成的 ACCEPTED CoT，不重新生成正文，不更改分段/guide/chunks；每段来自哪轮及审核日志均可查看。选取的是所有片段均通过自动审核的完整 episode，并不代表全量质量统计。\n\n"
        "正常 VLA 对象原样保留；Recovery 新生成目录没有独立 VLA，因此本页显式提供依据本轮 manifest 整理的动作目标，native 原始动作和状态在 trajectory.json / native_actions_by_segment.json。不能把它误认为旧 Recovery VLA。\n\n"
        "视频按 15 fps 便于回看，全部原始帧保留。状态时间戳独立展示（约 50 Hz）；视频回放时间不是原始物理时长。Recovery 另附失败→恢复完整配对视频，这些失败上下文不是学生输入。\n\n"
        "episodes/*/vlm.jsonl 保留源文件原始行字节与图片绝对路径；页面显示随包复制的图片，映射在 asset_mapping.json。model_audits 保存真实结果、双审及可追溯的请求记录。\n"
    )
    return data


def package(output):
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.relative_to(output).parts[0] not in {"work", "control"}:
                z.write(path, Path(output.name) / path.relative_to(output))
    with zipfile.ZipFile(output.with_suffix(".zip")) as z:
        if z.testzip() is not None:
            raise ValueError("archive integrity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    with writer_lock(output / "control/review.lock"):
        selected = sorted((output / "work/selected_inputs").glob("*.json"))
        if not selected and not args.render_only:
            raise ValueError("no frozen selected inputs")
        if args.render_only:
            episodes = [json.loads(p.read_text()) for p in sorted((output / "episodes").glob("*/review_episode.json"))]
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                episodes = list(pool.map(lambda p: build_episode(json.loads(p.read_text()), output), selected))
        render(output, episodes)
        package(output)
        print("review complete", output / "index.html", flush=True)


if __name__ == "__main__":
    main()
