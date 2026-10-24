"""Offline, interactive review of actual inspected episodes and 16D targets."""

import json
from dataclasses import fields
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from atomic_seg.pretrain.realomni.storage import atomic_bytes, atomic_json, read_json
from .config import Settings
from .contract import student_system, teacher_system, action_contract
from .media import read_video
from .runner import episode_dir, load_numeric, episode_from
from .source import iter_inventory


def build(run, output, limit=8):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "run_config.json")
    names = {field.name for field in fields(Settings)}
    settings = Settings(**{key: value for key, value in config["settings"].items() if key in names})
    examples = []
    for row in iter_inventory(run):
        directory = episode_dir(run, row)
        if not (directory / "status.json").exists():
            continue
        status = read_json(directory / "status.json")
        target = output / row["key"][:16]
        target.mkdir(exist_ok=True)
        item = {"id": row["key"][:16], "episode_id": row["episode_id"], "task": row["task"],
                "status": status["status"], "source_rows": row["length"],
                "reasons": status.get("reasons", []), "checks": status.get("checks", {}),
                "cameras": [], "segments": [], "model_labels": False}
        if (directory / "segmented").exists():
            result = read_json(directory / "segmented/segmentation.json")
            segments = result["segments"]
            if (directory / "annotated").exists():
                segments = read_json(directory / "annotated/segments.json")
                item["model_labels"] = True
            image_index = read_json(directory / "media/images.json") if (directory / "media/images.json").exists() else {}
            for s in segments:
                start, end = s["start_frame"], s["end_frame"]+1
                points = [start, (start+end)//2, end]
                shots = []
                for frame in points:
                    source = directory / "media/teacher" / f"{frame:07d}.jpg"
                    if not source.exists():
                        continue
                    destination = target / f"observation_{frame:07d}.jpg"
                    shutil.copyfile(source, destination)
                    shots.append({"frame": frame, "path": str(destination.relative_to(output))})
                item["segments"].append({**s, "shots": shots, "start_s": start/settings.fps, "end_s": end/settings.fps})
            for camera in settings.student_cameras:
                video = target / (camera + ".mp4")
                if not video.exists():
                    print("preview video", row["episode_id"], camera, flush=True)
                    read_video(row["videos"][camera], row["length"], settings.fps, settings.video_tolerance_s, preview=video)
                item["cameras"].append({"name": camera, "video": str(video.relative_to(output))})
            episode = episode_from(row, load_numeric(directory / "numeric"), settings)
            measured = np.asarray(episode["measured_states"])
            stride = max(1, len(measured)//900)
            item["signal"] = {"frames": list(range(0, len(measured), stride)),
                "left_gripper": measured[::stride, 7].tolist(), "right_gripper": measured[::stride, 15].tolist(),
                "left_xyz": measured[::stride, :3].tolist(), "right_xyz": measured[::stride, 8:11].tolist()}
            item["reference_rows"] = len(measured)-1
            atomic_json(target / "segmentation.json", result)
        decision = directory / ("rejected" if status["status"] == "rejected" else "checked") / "decision.json"
        if decision.exists():
            shutil.copyfile(decision, target / "decision.json")
        examples.append(item)
        if len(examples) >= limit:
            break
    examples.sort(key=lambda x: (not bool(x["segments"]), x["episode_id"]))
    data = {"run": str(run), "source": settings.source_root, "examples": examples,
            "contract": action_contract(settings), "student_system": student_system(settings),
            "teacher_system": teacher_system(settings), "version": config["version"]}
    atomic_json(output / "examples.json", data)
    template = Path(__file__).with_name("review.html").read_text()
    atomic_bytes(output / "index.html", template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")).encode())
    font = Path(__file__).resolve().parents[4] / "scripts/assets/agibot-review-cjk.woff"
    if font.exists():
        shutil.copyfile(font, output / "review.woff")
    return str(output / "index.html")
