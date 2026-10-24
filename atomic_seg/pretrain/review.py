"""Portable offline review using the existing episode review's visual language."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from atomic_seg.common.artifacts import atomic_write_json, utc_now
from atomic_seg.pretrain.pipeline import montage
from atomic_seg.pretrain.prompts import PROFILES, STUDENT_VERSION, student_system, teacher_system
from atomic_seg.pretrain.length_policy import length_statistics

ROOT = Path(__file__).resolve().parents[2]


def write_video(episode, destination):
    times = np.asarray(episode["timestamps"])
    ticks = np.rint(times * 1_000_000).astype(np.int64)
    dt = float(np.median(np.diff(times)))
    if destination.exists():
        with av.open(str(destination)) as check:
            if check.streams.video[0].frames == len(times):
                return
        raise ValueError("existing review video does not match full episode")
    temporary = destination.with_name("preview.tmp.mp4")
    first = montage(episode, 0)
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(1 / dt).limit_denominator(1000))
        stream.width, stream.height = first.size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = Fraction(1, 1_000_000)
        stream.options = {"crf": "23", "preset": "fast", "bf": "0"}
        for i, tick in enumerate(ticks):
            frame = av.VideoFrame.from_image(first if i == 0 else montage(episode, i))
            frame.pts = int(tick)
            frame.time_base = Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(temporary)) as check:
        if check.streams.video[0].frames != len(times):
            raise ValueError("review video dropped frames")
    temporary.replace(destination)


def build_review(config, episodes):
    output = Path(config["output_root"])
    report = json.loads((output / "VALIDATION.json").read_text())
    data = {"generated_at": utc_now(), "validation": report, "profiles": PROFILES, "episodes": []}
    from atomic_seg.prompt_contract import SINGLE_ARM_SYSTEM_PROMPT, MULTI_ARM_SYSTEM_PROMPT

    data["student_format"] = {
        "version": STUDENT_VERSION, "training_keys": ["messages", "images"],
        "shared_pipeline": "pretrain_noncot",
        "common_contract": ["next atomic subtask", "guide origin/endpoint and recomputation",
                            "inclusive eight-action chunks", "task arm constraints",
                            "compare previous command with observation and repeat/recover/advance", "four-field JSON"],
        "robotwin_reference_systems": {"single": SINGLE_ARM_SYSTEM_PROMPT, "multi": MULTI_ARM_SYSTEM_PROMPT},
        "dataset_systems": {p: student_system(p) for p in PROFILES},
        "metadata_path": "metadata/preview.jsonl",
    }
    atomic_write_json(output / "STUDENT_FORMAT.json", data["student_format"])
    for episode in episodes:
        destination = output / "episodes" / episode["id"]
        vla = json.loads((destination / "vla.json").read_text())
        vlm_text = (destination / "vlm.jsonl").read_text()
        rows = [json.loads(line) for line in vlm_text.splitlines()]
        metadata = [json.loads(line) for line in (destination / "vlm_metadata.jsonl").read_text().splitlines()]
        if len(rows) != len(metadata) or any(row["images"] != meta["images"] for row, meta in zip(rows, metadata)):
            raise ValueError("review metadata is not aligned with training rows")
        print("render video", episode["id"], len(episode["states"]), "frames", flush=True)
        write_video(episode, destination / "preview.mp4")
        stories = []
        for segment in vla["segments"]:
            start, end = segment["start_frame"], segment["end_frame"]
            frames = [start, (start + end) // 2, end]
            paths = []
            for frame in frames:
                path = destination / "storyboard" / f"{frame:06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    teacher_frame = output / "work" / episode["id"] / "teacher_frames" / f"{frame:06d}.jpg"
                    if teacher_frame.exists():
                        shutil.copyfile(teacher_frame, path)
                    else:
                        montage(episode, frame).save(path, quality=89)
                paths.append({"path": str(path.relative_to(output)), "frame": frame})
            stories.append(paths)
        segmentation = json.loads((output / "work" / episode["id"] / "segmentation.json").read_text())
        data["episodes"].append({**vla, "vlm": rows, "vlm_jsonl_text": vlm_text, "vlm_metadata": metadata,
            "observations": [m["review_images"] for m in metadata], "timestamps": episode["timestamps"],
            "directory": str(destination.relative_to(output)), "video": str((destination / "preview.mp4").relative_to(output)),
            "poster": stories[0][0]["path"], "storyboard": stories,
            "segmentation_method": segmentation["method"], "segmentation_config": segmentation["config"],
            "label_corrections": segmentation["label_corrections"],
            "length_audit": segmentation.get("length_audit"),
            "both_still_cleanup": segmentation.get("both_still_cleanup"),
            "core_still_refinement": segmentation.get("core_still_refinement"),
            "original_segments": segmentation.get("original_segments", segmentation["segments"]),
            "source_annotations": episode["source_annotations"], "alignment": episode["alignment"],
            "teacher_system": teacher_system(episode["profile"], grouped=episode.get("length_policy") is not None),
            "requests": [str(p.relative_to(output)) for p in sorted((output / "requests" / episode["id"]).glob("*.json"))]})
    grouped = [e for e in data["episodes"] if e["length_audit"]]
    soft_lengths = bool(grouped) and grouped[0]["length_audit"]["policy"].get("minimum_frames") is None
    if grouped:
        before = [s for e in grouped for s in e["original_segments"]]
        after = [s for e in grouped for s in e["segments"]]
        comparison = {
            "dataset": "ABC-130k", "policy": grouped[0]["length_audit"]["policy"],
            "before": length_statistics(before), "after": length_statistics(after),
            "episodes": [{"episode_id": e["id"], "task": e["task"], **e["length_audit"]} for e in grouped],
        }
        data["length_comparison"] = comparison
        atomic_write_json(output / "LENGTH_COMPARISON.json", comparison)
    if any(e.get("both_still_cleanup") for e in data["episodes"]):
        atomic_write_json(output / "BOTH_STILL_AUDIT.json", {
            e["id"]: {"adapter_cleanup": e["both_still_cleanup"], "core_cleanup": e["core_still_refinement"]}
            for e in data["episodes"] if e.get("both_still_cleanup")
        })
    atomic_write_json(output / "complete_episodes.json", data)
    template = (ROOT / "scripts" / "pretrain_episode_review.html").read_text()
    # Reuse the proven responsive style and offline CJK font of the earlier index.
    previous = (ROOT / "scripts" / "episode_dataset_review.html").read_text()
    style = previous.split("<style>", 1)[1].split("</style>", 1)[0]
    style = style.replace("fonts/review-cjk.woff2", "fonts/pretrain-review-cjk.woff").replace("format('woff2')", "format('woff')")
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__BASE_STYLE__", style).replace("__EPISODE_DATA__", embedded))
    (output / "fonts").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "scripts/assets/pretrain-review-cjk.woff", output / "fonts/pretrain-review-cjk.woff")
    shutil.copyfile(ROOT / "scripts/assets/OFL.txt", output / "fonts/OFL.txt")
    # Contact sheet uses real first/middle/final observations from all eight full episodes.
    sheet = Image.new("RGB", (1200, len(episodes) * 210 + 60), "#f2f6fc")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 18), "OXE / ABC-130k - complete episodes - non-CoT review", fill="#183556")
    for i, episode in enumerate(episodes):
        y = 60 + i * 210
        draw.text((15, y), f"{episode['id']} | {episode['task']}", fill="#183556")
        for j, frame in enumerate([0, len(episode["states"]) // 2, len(episode["states"]) - 1]):
            view = montage(episode, frame)
            view.thumbnail((390, 180))
            sheet.paste(view, (j * 400 + 5, y + 22))
    sheet.save(output / "overview.jpg", quality=92)
    (output / "README.md").write_text(
        "# OXE / ABC-130k 非 CoT 预训练数据预览\n\n"
        "解压后直接打开 index.html。页面包含完整视频、全部动作分段、原始 system/user/assistant、native VLA 动作、原始标注及 teacher prompt。\n\n"
        f"共 {len(episodes)} 个完整 episode，{report['planner_rows']} 条 planner 样本。OXE 本轮覆盖 Bridge 子集。ABC 同时覆盖有/无末端位姿、有/无原始 subtask 标注。\n\n"
        "学生输出仅含 guide_action、primary_action_verb、sub_task、num_chunks，无 CoT。Qwen 仅标注 sub_task；边界和数值由代码计算。\n\n"
        "训练行与 RoboTwin 对齐，仅含 messages/images；id、dataset、split、帧号和 image_roles 独立保存于 metadata/*.jsonl，按行对应。system 是一个完整段落，复用 RoboTwin 的 planner 规则；只替换真实机器人和坐标表示，移除 thinking 指令。\n\n"
        "user 按 RoboTwin 顺序组织图片标记、任务、视角、历史命令、执行行数、历史状态、当前状态及观测差分，以统一 no-thinking 指令结束。状态保留按臂命名的 xyz/rpy 或 joint_positions/gripper，并按 RoboTwin 方式显示四位小数；原始状态和动作数值不变。\n\n"
        "Bridge 保留 7 维 native action，guide 为末段命令目标减当前状态；ABC 保留 14 维 joint action，guide 为最后绝对关节命令减当前状态。不是 RoboTwin 固定 16 维合同。\n\n"
        "ABC 使用真实 action 时钟，当前/历史观测均不晚于对应 action 时间；完整视频按该时钟展示。Bridge 原始 RLDS 没有时间戳，5 Hz 为显式假定，帧号与原始动作行一一对应。\n\n"
        "源 train/val 身份保留，preview 汇总仅供查看；data/vlm、data/vla 下另有按来源 split 分开的文件。\n\n"
        "训练 images 使用实际存在的绝对路径，与 RoboTwin 格式一致；离线网页使用随包复制的相对路径图片，映射保存在独立 metadata，移动 ZIP 后页面仍可直接打开。\n\n"
        "VALIDATION.json 验证结构、全轨迹覆盖、动作原值、guide/chunks、图片与时间对齐；不表示每条语义标注都经人工验收。部分示范可能包含未完成动作，不把 task 文本或 is_terminal 自动解释为任务成功。\n\n"
        "requests/ 保留真实模型输入文本、图像路径及 hash、参数、原始响应；teacher_frames/ 保存实际送入模型的图像，供复核。\n"
        + (("\nABC 使用软长度偏好：倾向 32–64 个有效动作行，不设 16 帧硬下限。允许有必要保留的短动作，chunks 按真实动作行计算。双臂均静止的开头并入下一动作，中间并入上一动作；末尾短静止吸收，较长末尾保留一个 hold。吸收过程保留所有原始动作行，并记录在 BOTH_STILL_AUDIT.json。\n"
            if soft_lengths else "\nABC 使用配置中指定的硬下限和偏好长度。\n")
           + "仅合并左右臂动作标签分别相同的相邻片段，任一侧动作变化都保留边界；原始标注最近的既有动作边界也保留，偏移记录在 length_audit 中。LENGTH_COMPARISON.json 提供长度分布，primitive_segments 保留原始细分段。\n" if grouped else "")
        + "\nABC 的 14D 为左 6 关节角/左夹爪/右 6 关节角/右夹爪，关节角单位为弧度、base-to-wrist 排列；guide 为终点绝对关节命令减当前状态。关节 delta 不是末端 xyz，不能从正负号推断 Cartesian 抬升/下降；夹爪 delta=0 表示不变。学生 system 和当前 teacher 模板都明确这些含义，实际历史标注输入以 requests/ 原始日志为准。\n"
    )
    create_archive(output)
    print("review", str(output / "index.html"), flush=True)


def create_archive(output):
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        paths = []
        for parent, directories, files in os.walk(output):
            if Path(parent) == output:
                directories[:] = [d for d in directories if d not in {"work", "control", "logs"}]
            paths.extend(Path(parent) / name for name in files if not name.endswith(".log"))
        paths.extend(output.glob("work/*/teacher_frames/*.jpg"))
        for path in sorted(paths):
            relative = path.relative_to(output)
            archive.write(path, Path(output.name) / relative)
