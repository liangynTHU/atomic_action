"""Native preparation, non-CoT semantic annotation, aligned exports, and audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from atomic_seg.common.artifacts import atomic_write_json, atomic_write_jsonl, compact_json, sha256_file
from atomic_seg.common.model_client import multimodal_content, post_chat
from atomic_seg.pretrain.adapters import load_abc, load_bridge, tfrecord_examples
from atomic_seg.pretrain.prompts import ACTION_STARTS, VERSION, STUDENT_VERSION, student_system, teacher_system
from atomic_seg.pretrain.segmentation import segment_native, target_for
from atomic_seg.pretrain.student_format import build_user, cameras_in_order, json_text
from atomic_seg.pretrain.length_policy import validate_action_primitives


def digest(value):
    return hashlib.sha256(compact_json(value).encode()).hexdigest()


def prepare(config: dict):
    output = Path(config["output_root"])
    episodes = []
    for dataset in config["datasets"]:
        for choice in dataset["episodes"]:
            work = output / "work" / choice["id"]
            identity = digest({"choice": choice, "root": dataset["root"], "adapter": dataset["adapter"]})
            cache = work / "episode.json"
            if not cache.exists() and config.get("reuse_prepared_root"):
                reused = Path(config["reuse_prepared_root"]) / "work" / choice["id"] / "episode.json"
                if not reused.is_file():
                    raise FileNotFoundError(reused)
                episode = json.loads(reused.read_text())
                if episode.get("input_identity") != identity:
                    raise ValueError("reused prepared episode differs from requested source")
                episode["reused_preparation"] = {"path": str(reused), "sha256": sha256_file(reused)}
                atomic_write_json(cache, episode)
            if cache.exists():
                episode = json.loads(cache.read_text())
                if episode.get("input_identity", identity) != identity:
                    raise ValueError("source choice changed; use a new output_root")
                source = episode["source"]
                if source.get("sha256"):
                    if sha256_file(Path(source["file"])) != source["sha256"]:
                        raise ValueError("source MCAP changed since preparation")
                    annotation = Path(source["file"]).with_name("annotation.mcap")
                    current_hash = sha256_file(annotation) if annotation.exists() else None
                    if current_hash != source.get("annotation_sha256"):
                        raise ValueError("source annotations changed since preparation")
                elif source.get("record_sha256"):
                    for index, _, record_hash in tfrecord_examples(source["file"]):
                        if index == source["record_index"]:
                            if record_hash != source["record_sha256"]:
                                raise ValueError("source TFRecord changed since preparation")
                            break
                    else:
                        raise ValueError("source episode disappeared")
            else:
                print("decode", choice["id"], flush=True)
                loader = {"oxe_bridge": load_bridge, "abc_yam": load_abc}[dataset["adapter"]]
                episode = loader(Path(dataset["root"]), choice, work / "frames")
            episode.update(id=choice["id"], input_identity=identity)
            atomic_write_json(cache, episode)
            segmentation = segment_native(episode, length_policy=dataset.get("length_policy"))
            episode["length_policy"] = dataset.get("length_policy")
            atomic_write_json(cache, episode)
            atomic_write_json(work / "segmentation.json", segmentation)
            episodes.append(episode)
            print("prepared", episode["id"], len(episode["states"]), "rows", len(segmentation["segments"]), "segments", flush=True)
    return episodes


def load_prepared(config):
    return [json.loads((Path(config["output_root"]) / "work" / c["id"] / "episode.json").read_text())
            for d in config["datasets"] for c in d["episodes"]]


def montage(episode, frame, *, width=960):
    cameras = list(episode["camera_paths"])
    tile = width // len(cameras)
    height = max(240, int(tile * 0.75))
    image = Image.new("RGB", (width, height + 32), "#101a2c")
    draw = ImageDraw.Draw(image)
    for i, camera in enumerate(cameras):
        with Image.open(episode["camera_paths"][camera][frame]) as source:
            content = ImageOps.contain(source.convert("RGB"), (tile, height))
        image.paste(content, (i * tile + (tile - content.width) // 2, 32 + (height - content.height) // 2))
        draw.text((i * tile + 8, 9), f"{camera} | row {frame}", fill="white")
    return image


def label_batch(episode, segments, *, config, batch_index):
    output = Path(config["output_root"])
    work = output / "work" / episode["id"]
    images, roles, facts = [], [], []
    for segment in segments:
        start, end = segment["start_frame"], segment["end_frame"]
        frames = sorted(set([start, (start + end) // 2, end]))
        if segment.get("primitive_segments"):
            # Longer intervals need enough temporal coverage to label their whole motion.
            frames = sorted(set(np.linspace(start, end, 5, dtype=int).tolist()))
        for frame in frames:
            path = work / "teacher_frames" / f"{frame:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                montage(episode, frame).save(path, quality=90)
            images.append(path)
            roles.append(f"segment {segment['segment_index']}, source action row {frame}; chronological segment evidence")
        annotations = [a["text"] for a in episode["source_annotations"] if a["start_frame"] <= start]
        transitions = [a for a in episode["source_annotations"] if start < a["start_frame"] <= end]
        current = np.asarray(episode["states"][start])
        final = np.asarray(episode["states"][end])
        arms = ["left"] if episode["profile"] == "oxe_bridge" else ["left", "right"]
        starts = {("arm" if len(arms) == 1 else arm): ACTION_STARTS[segment[arm + "_action"]] for arm in arms}
        facts.append({**segment, "required_clause_starts": starts, "sampled_frames": frames, "measured_start_state": current.tolist(),
                      "measured_end_state": final.tolist(), "measured_state_delta": (final - current).tolist(),
                      "coarse_source_context": annotations[-1:], "source_transitions_in_interval": transitions})
    text = compact_json({"instruction": episode["task"], "segments": facts, "state_layout": episode["state_layout"],
                         "camera_order": list(episode["camera_paths"]), "output": "sub_task only, no reasoning"})
    system = teacher_system(episode["profile"], grouped=episode.get("length_policy") is not None)
    image_ids = [{"path": str(p), "sha256": sha256_file(p), "role": r} for p, r in zip(images, roles)]
    identity = digest({"system": system, "user": text, "images": image_ids, "model": config["model"], "version": VERSION})
    cache = work / "labels" / f"batch_{batch_index:04d}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["identity"] != identity:
            raise ValueError("teacher input changed; use a new output_root or explicitly remove stale label cache")
        validate_labels(saved["labels"], segments)
        return saved["labels"]["segments"]
    endpoints = [x.strip() for x in Path(config["endpoint_file"]).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    feedback = ""
    request_directory = output / "requests" / episode["id"]
    past_attempts = list(request_directory.glob(f"batch_{batch_index:04d}_attempt_*.json"))
    attempt_offset = max((int(p.stem.rsplit("_", 1)[1]) for p in past_attempts), default=-1) + 1
    for attempt in range(3):
        payload = {"model": config["model"], "temperature": 0.1, "max_tokens": 2200,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": multimodal_content(text + feedback, images, roles)}]}
        endpoint = endpoints[(batch_index + attempt + sum(episode["id"].encode())) % len(endpoints)]
        request = {"endpoint": endpoint, "identity": identity, "system": system, "user": text + feedback,
                   "images": image_ids, "parameters": {k: v for k, v in payload.items() if k != "messages"},
                   "payload_sha256": digest(payload)}
        log = request_directory / f"batch_{batch_index:04d}_attempt_{attempt_offset + attempt}.json"
        try:
            started = time.monotonic()
            response = post_chat(endpoint=endpoint, payload=payload, timeout=240, direct=True)
            request["elapsed_seconds"] = time.monotonic() - started
            request["response"] = response
            atomic_write_json(log, request)
            message = response["choices"][0]["message"]
            if message.get("reasoning_content") or message.get("reasoning"):
                raise ValueError("teacher returned hidden reasoning despite disabled thinking")
            labels = json.loads(message["content"])
            validate_labels(labels, segments)
            atomic_write_json(cache, {"identity": identity, "labels": labels, "request": str(log.relative_to(output))})
            print("labeled", episode["id"], batch_index, len(segments), flush=True)
            return labels["segments"]
        except Exception as error:
            request["error"] = str(error)
            atomic_write_json(log, request)
            feedback = "\nRetry: the previous response failed validation: " + str(error) + ". Follow the exact schema."
            if attempt == 2:
                raise


def validate_labels(labels, segments):
    if not isinstance(labels, dict) or set(labels) != {"segments"} or not isinstance(labels["segments"], list):
        raise ValueError("expected only a segments list")
    expected = [s["segment_index"] for s in segments]
    if [s.get("segment_index") for s in labels["segments"]] != expected:
        raise ValueError("segment indices must match requested indices exactly")
    for item, segment in zip(labels["segments"], segments):
        validate_action_primitives(segment)
        text = item.get("sub_task")
        if set(item) != {"segment_index", "sub_task"} or not isinstance(text, str) or not text.strip():
            raise ValueError("each label needs exactly segment_index and nonempty sub_task")
        word_limit = 70 if segment.get("primitive_segments") else 45
        if len(text.split()) > word_limit or re.search(r"<[^>]*>|\breasoning\b|\bbecause\b|\btherefore\b", text, re.I):
            raise ValueError("sub_task must be a concise action instruction without reasoning")
        if segment["primary_action_verb"].startswith("arm:"):
            if not text.startswith(ACTION_STARTS[segment["left_action"]]):
                raise ValueError("single-arm sub_task must start with its required action verb")
        else:
            match = re.fullmatch(r"Left arm: (.+)\. Right arm: (.+)\.", text)
            if not match:
                raise ValueError(f"segment {segment['segment_index']}: use exactly Left arm: <imperative>. Right arm: <imperative>.")
            for arm, clause in zip(["left", "right"], match.groups()):
                required = ACTION_STARTS[segment[arm + "_action"]]
                if not clause.startswith(required):
                    raise ValueError(f"segment {segment['segment_index']} {arm} arm clause must start exactly with '{required}' and describe that action")


def annotate(config, episodes):
    output = Path(config["output_root"])
    jobs = []
    all_segments = {}
    for episode in episodes:
        segmentation = json.loads((output / "work" / episode["id"] / "segmentation.json").read_text())
        segments = segmentation["segments"]
        for segment in segments:
            validate_action_primitives(segment)
        if config.get("reuse_annotations_root") or (config.get("reuse_prepared_root") and not episode.get("length_policy")):
            source = Path(config.get("reuse_annotations_root") or config["reuse_prepared_root"])
            original_episode = json.loads((source / "work" / episode["id"] / "episode.json").read_text())
            for key in ("source", "states", "actions", "command_endpoints", "task", "profile", "source_annotations"):
                if original_episode[key] != episode[key]:
                    raise ValueError(f"annotation reuse source differs: {key}")
            old_segments = json.loads((source / "work" / episode["id"] / "annotated_segments.json").read_text())
            compared_keys = ("start_frame", "end_frame", "left_action", "right_action", "primary_action_verb")
            if len(segments) != len(old_segments) or any(
                any(a[k] != b[k] for k in compared_keys) for a, b in zip(segments, old_segments)
            ):
                raise ValueError("unchanged-source label reuse requires identical boundaries and actions")
            # Reuse text only. Keep this run's grouping/still metadata and boundaries.
            all_segments[episode["id"]] = [
                {**segment, "sub_task": old["sub_task"]}
                for segment, old in zip(segments, old_segments)
            ]
            shutil.copytree(source / "requests" / episode["id"], output / "requests" / episode["id"], dirs_exist_ok=True)
            shutil.copytree(source / "work" / episode["id"] / "teacher_frames",
                            output / "work" / episode["id"] / "teacher_frames", dirs_exist_ok=True)
            atomic_write_json(output / "work" / episode["id"] / "label_reuse.json", {
                "source": str(source), "same_boundaries_and_actions": True,
                "same_states_task_and_source_annotations": True,
                "reason": "unchanged source, boundaries and action labels; reuse semantic text and retain current segmentation metadata",
                "annotation_sha256": sha256_file(source / "work" / episode["id"] / "annotated_segments.json"),
            })
            continue
        all_segments[episode["id"]] = segments
        size = config.get("batch_size", 4)
        jobs.extend((episode, segments[i:i + size], i // size) for i in range(0, len(segments), size))
    with ThreadPoolExecutor(max_workers=config.get("workers", 4)) as executor:
        futures = {executor.submit(label_batch, e, s, config=config, batch_index=i): e["id"] for e, s, i in jobs}
        for future in as_completed(futures):
            for item in future.result():
                all_segments[futures[future]][item["segment_index"]]["sub_task"] = item["sub_task"].strip()
    for episode in episodes:
        atomic_write_json(output / "work" / episode["id"] / "annotated_segments.json", all_segments[episode["id"]])


def export_episode(episode, segments, output):
    for segment in segments:
        validate_action_primitives(segment)
    directory = Path("episodes") / episode["id"]
    destination = output / directory
    destination.mkdir(parents=True, exist_ok=True)
    rows, metadata, vla, previous = [], [], [], None
    for segment in segments:
        start, end = segment["start_frame"], segment["end_frame"]
        target = target_for(episode, segment)
        user, image_roles, frames, execution = build_user(episode, segment, previous)
        image_paths, relative_paths, source_paths = [], [], []
        for when, frame in frames:
            for camera in cameras_in_order(episode):
                paths = episode["camera_paths"][camera]
                relative = directory / "observations" / f"{camera}_{frame:06d}.jpg"
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                if not (output / relative).exists():
                    shutil.copyfile(paths[frame], output / relative)
                image_paths.append(str((output / relative).resolve()))
                relative_paths.append(str(relative))
                source_paths.append(paths[frame])
        row = {"messages": [{"role": "system", "content": student_system(episode["profile"])},
                            {"role": "user", "content": user}, {"role": "assistant", "content": json_text(target)}],
               "images": image_paths}
        rows.append(row)
        count = end - start + 1
        metadata.append({"schema": "pretrain_noncot_metadata", "prompt_version": STUDENT_VERSION,
                         "id": f"{episode['id']}_{segment['segment_index']:04d}",
                         "episode_id": episode["id"], "dataset": episode["dataset"],
                         "source_split": episode["source"]["split"], "profile": episode["profile"],
                         "segment_index": segment["segment_index"], "start_frame": start, "end_frame": end,
                         "image_roles": image_roles, "images": image_paths, "review_images": relative_paths,
                         "source_images": source_paths, "target": target, "state_layout": episode["state_layout"],
                         "current_state": episode["states"][start], "guide_action_target_state": episode["command_endpoints"][end],
                         "previous_boundary_frame": previous["start_frame"] if previous else None,
                         "previous_execution": execution, "valid_action_rows": count,
                         "aligned_action_slots": target["num_chunks"] * 8,
                         "padded_action_slots": target["num_chunks"] * 8 - count})
        vla.append({**segment, **target, "state_at_start": episode["states"][start],
                    "command_endpoint": episode["command_endpoints"][end],
                    "actions": episode["actions"][start:end + 1], "timestamps": episode["timestamps"][start:end + 1],
                    "valid_action_rows": count, "padding_rows_if_chunked": 8 * target["num_chunks"] - count})
        previous = segment
    record = {k: episode[k] for k in ["id", "dataset", "subset", "profile", "task", "source", "state_layout", "action_layout", "time_basis"]}
    record.update(frame_count=len(episode["states"]), camera_names=list(episode["camera_paths"]), segments=vla,
                  action_semantics=("reference actions from adjacent measured end-effector states"
                                    if episode["profile"] == "realomni_umi" else "native actions")
                  + "; no padding stored; use valid_action_rows when packing 8-action chunks")
    trajectory = {k: episode[k] for k in ["states", "actions", "command_endpoints", "timestamps", "native_aux", "alignment", "source_annotations", "end_effector_poses"]}
    if episode["profile"] == "realomni_umi":
        record["action_contract"] = episode["action_contract"]
        trajectory.update({k: episode[k] for k in ("measured_states", "measured_timestamps_ns", "action_contract")})
    atomic_write_jsonl(destination / "vlm.jsonl", rows)
    atomic_write_jsonl(destination / "vlm_metadata.jsonl", metadata)
    atomic_write_json(destination / "vla.json", record)
    atomic_write_json(destination / "trajectory.json", trajectory)
    atomic_write_json(destination / "segmentation.json", segments)
    return rows, record


def audit_episode(episode, rows, vla, output):
    n = len(episode["states"])
    cursor = 0
    reconstructed = []
    dim = len(episode["states"][0])
    if episode.get("timestamps_ns"):
        clock = np.asarray(episode["timestamps_ns"], dtype=np.int64)
        for name, alignment in episode["alignment"]["streams"].items():
            aligned = np.asarray(alignment["source_timestamps_ns"], dtype=np.int64)
            if episode["profile"] == "realomni_umi":
                if len(aligned) != n + 1:
                    raise ValueError("full observation clock length mismatch")
                aligned = aligned[:-1]
            if len(aligned) != n:
                raise ValueError("source clock length mismatch")
            if alignment["method"] == "latest_at_or_before" and np.any(aligned > clock):
                raise ValueError(f"future observation leaked: {name}")
    for index, (row, segment) in enumerate(zip(rows, vla["segments"])):
        validate_action_primitives(segment)
        if segment["start_frame"] != cursor or segment["end_frame"] < cursor:
            raise ValueError("gapped or overlapping segments")
        cursor = segment["end_frame"] + 1
        hard_minimum = (episode.get("length_policy") or {}).get("minimum_frames")
        if hard_minimum is not None and len(segment["actions"]) < hard_minimum:
            raise ValueError("segment is below the hard minimum of valid action rows")
        if episode["profile"] in {"abc_yam", "realomni_umi"} and index < len(vla["segments"]) - 1 and segment["left_action"] == segment["right_action"] == "still":
            raise ValueError("bimanual episode has a standalone initial/intermediate both-still target")
        target = json.loads(row["messages"][-1]["content"])
        if set(row) != {"messages", "images"}:
            raise ValueError("training rows must contain only messages and images")
        if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
            raise ValueError("unexpected student conversation roles")
        if row["messages"][0]["content"] != student_system(episode["profile"]):
            raise ValueError("student system differs from shared prompt contract")
        previous = vla["segments"][index - 1] if index else None
        expected_user, roles, _, execution = build_user(episode, segment, previous)
        if row["messages"][1]["content"] != expected_user:
            raise ValueError("student user/history differs from source states and previous command")
        if set(target) != {"guide_action", "primary_action_verb", "sub_task", "num_chunks"}:
            raise ValueError("unexpected assistant keys")
        if any(x in row["messages"][-1]["content"].lower() for x in ["<think", "</think", '"reasoning"', '"cot"']):
            raise ValueError("CoT leaked into student output")
        expected = target_for(episode, segment)
        if target != expected or len(target["guide_action"]) != dim or not np.isfinite(target["guide_action"]).all():
            raise ValueError("guide_action/label mismatch")
        if target["num_chunks"] != math.ceil(len(segment["actions"]) / 8):
            raise ValueError("invalid num_chunks")
        if row["messages"][1]["content"].count("<image>") != len(row["images"]):
            raise ValueError("image token mismatch")
        if len(row["images"]) != len(roles):
            raise ValueError("actual camera count differs from student image roles")
        for image in row["images"]:
            with Image.open(output / image) as im:
                im.verify()
        previous_start = previous["start_frame"] if previous else None
        expected_frames = ([] if previous_start is None else [previous_start]) + [segment["start_frame"]]
        image_index = 0
        for frame in expected_frames:
            for camera in cameras_in_order(episode):
                paths = episode["camera_paths"][camera]
                if sha256_file(output / row["images"][image_index]) != sha256_file(Path(paths[frame])):
                    raise ValueError(f"student observation provenance mismatch: {camera}")
                image_index += 1
        reconstructed.extend(segment["actions"])
    if cursor != n or reconstructed != episode["actions"] or len(rows) != len(vla["segments"]):
        raise ValueError("complete native action trajectory was not preserved")
    if episode["profile"] == "realomni_umi":
        measured = np.asarray(episode["measured_states"])
        if (measured.shape != (n + 1, 16) or not np.array_equal(measured[:-1], episode["states"])
                or not np.array_equal(measured[1:], episode["actions"])
                or not np.array_equal(measured[1:], episode["command_endpoints"])):
            raise ValueError("reference trajectory differs from adjacent measured states")
    return {"episode_id": episode["id"], "dataset": episode["dataset"], "status": "PASS",
            "source_action_rows": n, "exported_action_rows": len(reconstructed), "segments": len(rows),
            "state_dim": dim, "guide_dim": dim, "no_cot": True, "full_coverage": True,
            "native_actions_unchanged": episode["profile"] != "realomni_umi",
            "reference_actions_verified": episode["profile"] == "realomni_umi", "guide_and_chunks_recomputed": True,
            "student_image_provenance_verified": True,
            "robotwin_aligned_student_format": True, "history_execution_verified": True,
            "both_still_policy_verified": episode["profile"] in {"abc_yam", "realomni_umi"},
            "length_constraint": (
                "not_applied" if episode.get("length_policy") is None else
                "soft_preference" if episode["length_policy"].get("minimum_frames") is None else "explicit_hard_minimum"
            ),
            "causal_observations": True, "source_split": episode["source"]["split"],
            "max_alignment_skew_ms": max((v["max_abs_skew_ms"] for v in episode["alignment"].get("streams", {}).values()), default=0)}


def export(config, episodes):
    output = Path(config["output_root"])
    all_rows, all_metadata, all_vla, checks = [], [], {}, []
    for episode in episodes:
        segments = json.loads((output / "work" / episode["id"] / "annotated_segments.json").read_text())
        rows, vla = export_episode(episode, segments, output)
        checks.append(audit_episode(episode, rows, vla, output))
        all_rows.extend(rows)
        metadata_path = output / "episodes" / episode["id"] / "vlm_metadata.jsonl"
        all_metadata.extend(json.loads(line) for line in metadata_path.read_text().splitlines())
        all_vla[episode["id"]] = vla
    # Pilot container intentionally preserves train/val identities; no implicit split mixing.
    atomic_write_jsonl(output / "data" / "vlm" / "preview.jsonl", all_rows)
    atomic_write_jsonl(output / "metadata" / "preview.jsonl", all_metadata)
    atomic_write_json(output / "data" / "vla" / "preview.json", all_vla)
    for split in sorted({m["source_split"] for m in all_metadata}):
        atomic_write_jsonl(output / "data" / "vlm" / f"{split}.jsonl", [r for r, m in zip(all_rows, all_metadata) if m["source_split"] == split])
        atomic_write_jsonl(output / "metadata" / f"{split}.jsonl", [m for m in all_metadata if m["source_split"] == split])
        atomic_write_json(output / "data" / "vla" / f"{split}.json", {k: v for k, v in all_vla.items() if v["source"]["split"] == split})
    report = {"status": "PASS", "prompt_version": STUDENT_VERSION, "teacher_version": VERSION, "episodes": checks,
              "episode_count": len(episodes), "planner_rows": len(all_rows),
              "model": config["model"], "teacher_role": "sub_task annotation only, thinking disabled",
              "scope": "review pilot; OXE coverage is Bridge only; semantic labels and boundaries await human review"}
    from atomic_seg.pretrain.length_policy import length_statistics

    report["length_statistics"] = {dataset: length_statistics([
        s for episode in episodes if episode["dataset"] == dataset for s in all_vla[episode["id"]]["segments"]
    ]) for dataset in sorted({e["dataset"] for e in episodes})}
    atomic_write_json(output / "VALIDATION.json", report)
    return report
