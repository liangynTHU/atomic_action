#!/usr/bin/env python3
"""Build an offline, source-backed RealOmni quality and segmentation review."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
from PIL import Image, ImageDraw
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from atomic_seg.pretrain.realomni.storage import (
    atomic_bytes, atomic_json, check_source, digest, episode_dir, file_hash,
    iter_jsonl, read_json, verify_directory,
)


LABELS = {
    "clean_bowl": "清洗碗具", "fold_and_store_clothes": "折叠并收纳衣物",
    "lace_up_shoes_with_both_hands": "双手系鞋带", "desktop_object_sorting": "桌面物品整理",
    "Clutter-Tidy-Up-[Stage2]": "杂物整理 · Stage 2", "fold_and_store_large_clothes": "大件衣物整理",
    "organize_scattered_shoes": "整理鞋子", "clean_container": "清洗容器",
    "unscrew_bottle_cap_and_pour": "开瓶盖与倾倒",
}


def title_for(episode_id):
    return next((title for key, title in LABELS.items() if key in Path(episode_id).parts), Path(episode_id).parent.name)


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_page(output, data):
    template = Path(__file__).with_name("realomni_review.html").read_text()
    names = ["examples.json", "all_results.jsonl"]
    for e in data["examples"]:
        names.extend([e["decision_file"], e["id"] + "/example.json"])
        if e.get("segmentation"):
            names.append(e["id"] + "/segmentation.json")
    # Preserve nanosecond integers exactly: JSON.parse/JSON.stringify in the
    # browser cannot round-trip integers above Number.MAX_SAFE_INTEGER.
    originals = {name: (output / name).read_text() for name in names}
    data_text = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    raw_text = json.dumps(originals, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__REVIEW_DATA__", data_text).replace("__DOWNLOAD_TEXTS__", raw_text)
    atomic_bytes(output / "index.html", page.encode())


def extract_media(source, target, audit):
    """Decode full source cameras for review only; never change the quality decision."""
    target.mkdir(parents=True, exist_ok=True)
    sources = audit.get("streams", {})
    contexts, outputs, cameras, stamps = {}, {}, {}, {topic: [] for topic in sources}
    posters, sample_points, codecs, errors = {}, {}, {}, {}
    for robot, name in (("robot0", "left_wrist"), ("robot1", "right_wrist")):
        topic = f"/{robot}/sensor/camera0/compressed"
        if topic not in sources or not sources[topic].get("count"):
            continue
        expected = sources[topic]["count"]
        sample_points[topic] = {0: "start", (expected - 1) // 2: "middle", expected - 1: "end"}
        cameras[topic] = {"name": name, "count": 0, "expected_frames": expected,
                          "first_timestamp_ns": sources[topic]["first_timestamp_ns"], "storyboard": []}

    def accept(topic, frames):
        camera = cameras[topic]
        for frame in frames:
            if frame.pts is None or frame.is_corrupt:
                raise ValueError(f"source has corrupt or untimestamped review frame: {topic}")
            stamp = int(frame.pts)
            if camera["count"] and stamp <= camera["last_timestamp_ns"]:
                raise ValueError(f"non-increasing decoded timestamps: {topic}")
            if topic not in outputs:
                destination = target / (camera["name"] + ".mp4")
                container = av.open(str(destination), "w", options={"movflags": "+faststart"})
                stream = container.add_stream("libx264", rate=30)
                width = min(480, frame.width)
                height = max(2, round(frame.height * width / frame.width / 2) * 2)
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                stream.codec_context.thread_count = 2
                stream.time_base = Fraction(1, 90_000)
                stream.codec_context.time_base = Fraction(1, 90_000)
                stream.options = {"preset": "veryfast", "crf": "25"}
                outputs[topic] = (container, stream)
                camera["video"] = str(destination.name)
            container, stream = outputs[topic]
            image = frame.reformat(width=stream.width, height=stream.height, format="rgb24").to_image()
            index = camera["count"]
            if index in sample_points[topic]:
                relative = f"{camera['name']}_{sample_points[topic][index]}.jpg"
                image.save(target / relative, quality=90)
                camera["storyboard"].append({"path": relative, "frame": index, "timestamp_ns": stamp,
                                              "role": sample_points[topic][index]})
                if index == 0:
                    posters[camera["name"]] = image.copy()
                    camera["poster"] = relative
            output_frame = av.VideoFrame.from_image(image)
            output_frame.pts = round((stamp - camera["first_timestamp_ns"]) * 90_000 / 1e9)
            output_frame.time_base = Fraction(1, 90_000)
            for packet in stream.encode(output_frame):
                container.mux(packet)
            camera["count"] += 1
            camera["last_timestamp_ns"] = stamp

    try:
        with Path(source).open("rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()], validate_crcs=True)
            for _, channel, _, message in reader.iter_decoded_messages(topics=list(sources), log_time_order=False):
                topic = channel.topic
                stamp = int(message.header.timestamp)
                stamps[topic].append(stamp)
                if topic not in cameras or topic in errors:
                    continue
                if topic not in contexts:
                    codec = message.format
                    if codec not in {"h264", "h265", "hevc"}:
                        raise ValueError(f"unsupported source codec: {codec}")
                    contexts[topic] = av.CodecContext.create("hevc" if codec == "h265" else codec, "r")
                    contexts[topic].thread_count = 2
                    contexts[topic].options = {"err_detect": "explode"}
                    codecs[topic] = codec
                if codecs[topic] != message.format:
                    raise ValueError("source codec changed")
                packet = av.Packet(bytes(message.data))
                packet.pts, packet.time_base = stamp, Fraction(1, 1_000_000_000)
                try:
                    accept(topic, contexts[topic].decode(packet))
                except (av.FFmpegError, ValueError) as error:
                    errors[topic] = {"topic": topic, "error": str(error), "timestamp_ns": stamp,
                                     "decoded_before_error": cameras[topic]["count"]}
        for topic, context in contexts.items():
            if topic in errors:
                continue
            try:
                accept(topic, context.decode(None))
            except (av.FFmpegError, ValueError) as error:
                errors[topic] = {"topic": topic, "error": str(error), "decoded_before_error": cameras[topic]["count"]}
        for topic, (container, stream) in outputs.items():
            if topic in errors:
                continue
            for packet in stream.encode(None):
                container.mux(packet)
            if cameras[topic]["count"] != sources[topic]["count"]:
                errors[topic] = {"topic": topic, "error": "decoded review count differs from source count",
                                 "decoded": cameras[topic]["count"], "expected": sources[topic]["count"]}
    finally:
        for container, _ in outputs.values():
            container.close()
    if errors:
        # Do not publish playable partial videos as a complete episode preview.
        for p in target.glob("*.mp4"):
            p.rename(p.with_suffix(".incomplete.mp4"))
        return {}, stamps, {}, list(errors.values())
    return cameras, stamps, posters, []


def coverage_data(audit, stamps, settings):
    streams = audit.get("streams", {})
    available = [s for s in streams.values() if s.get("count", 0) >= 2]
    if not available:
        return [], None, None
    start = min(s["first_timestamp_ns"] for s in available)
    end = max(s["last_timestamp_ns"] for s in available)
    rows = []
    for topic, stream in streams.items():
        reasons = [r for r in audit["reasons"] if r.get("topic") == topic]
        if stream.get("count", 0) < 2:
            rows.append({"topic": topic, "count": stream.get("count", 0), "missing": True, "reasons": reasons, "gaps": []})
            continue
        gaps = []
        for reason in reasons:
            if reason["code"] != "timestamp_gap_or_disorder":
                continue
            source_stamps = stamps.get(topic, [])
            for i in reason.get("first_indices", []):
                if i + 1 < len(source_stamps):
                    gaps.append({"after_frame": i, "at_s": (source_stamps[i] - start) / 1e9,
                                 "duration_ms": (source_stamps[i + 1] - source_stamps[i]) / 1e6})
        rows.append({"topic": topic, "count": stream["count"], "hz": stream["expected_hz"],
                     "start_s": (stream["first_timestamp_ns"] - start) / 1e9,
                     "end_s": (stream["last_timestamp_ns"] - start) / 1e9,
                     "prefix_ms": (stream["first_timestamp_ns"] - start) / 1e6,
                     "suffix_ms": (end - stream["last_timestamp_ns"]) / 1e6,
                     "gap_count": stream["gap_count"], "max_gap_ms": stream["maximum_gap_ns"] / 1e6,
                     "gap_limit_ms": settings["maximum_gap_periods"] * 1000 / stream["expected_hz"],
                     "boundary_limit_ms": settings["maximum_boundary_skew_periods"] * 1000 / stream["expected_hz"],
                     "gaps": gaps, "reasons": reasons})
    return rows, start, end


def reuse_example(source, target, name, summary, audit):
    path = source / name / "example.json"
    if not path.exists():
        return None
    previous = read_json(path)
    if (previous["source_sha256"] != summary["source_sha256"] or previous["episode_id"] != summary["episode_id"]
            or previous["audit"] != audit or not previous["cameras"]):
        return None
    files = []
    posters = {}
    for camera in previous["cameras"]:
        video = source / camera["video"]
        with av.open(str(video)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            count = sum(1 for _ in container.decode(video=0))
        if count != camera["count"]:
            raise ValueError("cached review video frame count changed")
        files.append(video)
        for item in camera["storyboard"]:
            image = source / item["path"]
            with Image.open(image) as im:
                im.verify()
            files.append(image)
        with Image.open(source / camera["poster"]) as im:
            posters[camera["name"]] = im.copy()
    for file in files:
        if file.resolve() != (target / file.name).resolve():
            shutil.copyfile(file, target / file.name)
    segmentation_file = source / name / "segmentation.json"
    if previous.get("segmentation") and segmentation_file.resolve() != (target / "segmentation.json").resolve():
        shutil.copyfile(segmentation_file, target / "segmentation.json")
    previous.update(summary)
    previous["reused_review_media"] = str(path)
    previous["media_errors"] = []
    return previous, posters


def build_review(runs, output, limit=6, reuse_media_from=None):
    output = output.resolve()
    identity = {"runs": [str(p) for p in runs], "limit": limit,
                "run_versions": [read_json(p / "run_config.json")["version"] for p in runs],
                "builder": file_hash(Path(__file__)), "template": file_hash(Path(__file__).with_name("realomni_review.html"))}
    marker = output / "review_build.json"
    if output.exists() and any(output.iterdir()):
        if not marker.exists() or read_json(marker) != identity:
            raise ValueError("review output belongs to a different build; choose a new output directory")
    for run in runs:
        source = Path(read_json(run / "run_config.json")["settings"]["source_root"]).resolve()
        if output == source or source in output.parents or output == run or output in run.parents:
            raise ValueError("review output must be separate from source and run data")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, identity)
    all_rows, examples, posters = [], [], []
    seen = set()
    for run in runs:
        config = read_json(run / "run_config.json")
        for row in iter_jsonl(run / "inventory/episodes.jsonl"):
            if row["episode_id"] in seen:
                continue
            seen.add(row["episode_id"])
            directory = episode_dir(run, row)
            state = read_json(directory / "status.json") if (directory / "status.json").exists() else {"status": "pending"}
            checked = directory / "checked"
            if not checked.exists():
                all_rows.append({"episode_id": row["episode_id"], "status": state["status"]})
                continue
            verify_directory(checked, digest({"version": config["version"], "source": row["source_identity"], "stage": "checked"}))
            decision = read_json(checked / "decision.json")
            summary = {"episode_id": row["episode_id"], "status": state["status"],
                       "decision": decision["decision"], "reasons": decision["audit"]["reasons"],
                       "source_file": row["file"]["path"], "source_sha256": decision["source_sha256"],
                       "source_run": str(run), "segments": state.get("segments", 0)}
            all_rows.append(summary)
            if len(examples) >= limit:
                continue
            check_source(row)
            if file_hash(row["file"]["path"]) != decision["source_sha256"]:
                raise ValueError("source changed since quality decision")
            name = f"example_{len(examples) + 1:02d}"
            target = output / name
            target.mkdir(exist_ok=True)
            save_json(target / "decision.json", decision)
            save_json(target / "status.json", state)
            cached = reuse_example(reuse_media_from or output, target, name, summary, decision["audit"])
            if cached:
                example, images = cached
                examples.append(example)
                posters.append((example, images))
                save_json(target / "example.json", example)
                print("reused validated review media", row["episode_id"], flush=True)
                continue
            print("review media", row["episode_id"], flush=True)
            cameras, stamps, images, media_errors = extract_media(row["file"]["path"], target, decision["audit"])
            check_source(row)
            if file_hash(row["file"]["path"]) != decision["source_sha256"]:
                raise ValueError("source changed during media extraction")
            streams, start, end = coverage_data(decision["audit"], stamps, config["settings"])
            for camera in cameras.values():
                camera["offset_s"] = (camera["first_timestamp_ns"] - start) / 1e9
                camera["end_s"] = (camera["last_timestamp_ns"] - start) / 1e9
                for key in ("video", "poster"):
                    camera[key] = f"{name}/{camera[key]}"
                for image in camera["storyboard"]:
                    image["path"] = f"{name}/{image['path']}"
                    image["time_s"] = (image["timestamp_ns"] - start) / 1e9
            segmentation = None
            if state["status"] == "segmented":
                verify_directory(directory / "segmented", digest({"version": config["version"], "source": row["source_identity"], "stage": "segmented"}))
                segmentation = read_json(directory / "segmented/segmentation.json")
                save_json(target / "segmentation.json", segmentation)
            example = {**summary, "id": name, "title": title_for(row["episode_id"]),
                       "duration_s": (end - start) / 1e9 if start is not None else 0,
                       "audit": decision["audit"], "cameras": list(cameras.values()), "streams": streams,
                       "media_errors": media_errors,
                       "decision_file": f"{name}/decision.json", "segmentation": segmentation,
                       "display_only": "full source video previews; no rejected trajectory was segmented or accepted"}
            examples.append(example)
            posters.append((example, images))
            save_json(target / "example.json", example)
    data = {"schema": "realomni_source_review_v1", "summary": dict(Counter(x["status"] for x in all_rows)),
            "inspected_episodes": len(all_rows), "preview_episodes": len(examples), "examples": examples,
            "base_filter_enabled": False, "all_results": all_rows}
    save_json(output / "examples.json", data)
    atomic_bytes(output / "all_results.jsonl", ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_rows)).encode())
    write_page(output, data)
    font = Path(__file__).parent / "assets/realomni-review-cjk.woff"
    shutil.copyfile(font, output / "review-cjk.woff")
    shutil.copyfile(font.with_name("OFL.txt"), output / "OFL.txt")
    canvas = Image.new("RGB", (1000, len(posters) * 250), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    for i, (example, images) in enumerate(posters):
        y = i * 250
        draw.text((18, y + 12), f"{example['id']} | {example['status']} | {Path(example['episode_id']).name}", fill="#26364a")
        for j, side in enumerate(("left_wrist", "right_wrist")):
            im = images.get(side)
            if im:
                im.thumbnail((470, 204))
                canvas.paste(im, (18 + j * 490, y + 36))
    canvas.save(output / "overview.jpg", quality=90)
    summaries = ["# RealOmni 真实检查结果", "", f"共检查 {len(all_rows)} 条，本页展示 {len(examples)} 条。", "",
                 "底盘判断关闭。被拒绝的 episode 未生成分割和训练样本。预览视频仅供核对源数据。", "",
                 "| 示例 | 源文件 | 结果 |", "|---|---|---|"]
    for e in examples:
        summaries.append(f"| {e['id']} · {e['title']} | `{e['episode_id']}` | {e['status']} |")
    summaries += ["", "打开 index.html 查看完整双路视频、时间覆盖、缺口证据和原始判定 JSON。",
                  "all_results.jsonl 含所有已检查样本；examples.json 含本页完整结构。"]
    atomic_bytes(output / "README.md", ("\n".join(summaries) + "\n").encode())
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--reuse-media-from", type=Path, help="reuse complete source-matching previews after frame-count validation")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    result = build_review([p.resolve() for p in args.run_root], args.output_root, args.limit,
                          args.reuse_media_from.resolve() if args.reuse_media_from else None)
    print(json.dumps({"index": str(args.output_root.resolve() / "index.html"), "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
